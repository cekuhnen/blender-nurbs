"""Sidebar panels (active object, analyze, settings), scene settings. The tool
panels and menus are in toolgroups.py."""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty

from . import features, store
from .kernel import load as load_kernel


def _retess(self, context):
    for o in context.scene.objects:
        if store.is_nurbs(o):
            try:
                store.retessellate(o, context)
            except Exception as exc:
                print("NURBS retessellate:", o.name, exc)


def _redraw_views(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _rd(self, context):
    _redraw_views(context)


_env_items = []


def _env_source_items(self, context):
    """Light Tunnel, Horizon, then Blender's bundled HDRIs (the list is kept
    alive here: Blender doesn't copy a dynamic enum's strings)."""
    if not _env_items:
        from . import environments as env
        _env_items.extend(env.PROCEDURAL)
        for n in env.hdri_names():
            _env_items.append((n, n.replace("_", " ").title(), f"Blender's {n} HDRI (1K, CC0, Poly Haven)"))
    return _env_items


def _one_map(settings, which):
    """Zebra, Draft Angle and Curvature Map are exclusive (one surface overlay)."""
    if getattr(settings, which):
        for other in ("show_zebra", "show_draft", "show_curvature", "show_matcap"):
            if other != which and getattr(settings, other):
                setattr(settings, other, False)
    _redraw_views(bpy.context)


def _on_pause(self, context):
    """Pause Rebuild switched off: build everything that waited."""
    if self.pause_rebuild:
        _redraw_views(context)
        return
    from . import history
    n, errors = history.rebuild_waiting(context)
    for e in errors:
        print("NURBS rebuild after pause:", e)
    _redraw_views(context)


class NPROTO_OT_rebuild_waiting(bpy.types.Operator):
    """Build every surface whose settings changed while Pause Rebuild was on (and what is built from it)"""
    bl_idname = "nproto.rebuild_waiting"
    bl_label = "Rebuild"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import history
        n, errors = history.rebuild_waiting(context)
        _redraw_views(context)
        if errors:
            self.report({'ERROR'}, "Rebuild failed: " + "; ".join(errors))
            return {'FINISHED'} if n > len(errors) else {'CANCELLED'}
        self.report({'INFO'}, f"Rebuilt {n} object{'s' if n != 1 else ''}" if n else "Nothing was waiting")
        return {'FINISHED'}


def draw_pause(layout, context):
    """Pause Rebuild toggle and, while it holds anything back, the Rebuild button."""
    from . import history
    s = context.scene.nproto_settings
    row = layout.row(align=True)
    row.prop(s, "pause_rebuild", text="Paused" if s.pause_rebuild else "Pause Rebuild",
             icon='PAUSE', toggle=True)
    if s.pause_rebuild:
        n = len(history.waiting(context.scene))
        sub = row.row(align=True)
        sub.alert = n > 0
        sub.enabled = n > 0
        sub.operator("nproto.rebuild_waiting", text=f"Rebuild ({n})" if n else "Rebuild", icon='FILE_REFRESH')


class NprotoSettings(bpy.types.PropertyGroup):
    pause_rebuild: BoolProperty(
        name="Pause Rebuild", default=False, update=_on_pause,
        description="Hold rebuilds back: change continuity (labels), tension and other feature settings "
                    "without waiting for each solve, then press Rebuild (or switch the pause off)")
    construction_history: BoolProperty(
        name="Construction History", default=True,
        description="Record how surfaces were built so they follow their inputs")
    live_update: BoolProperty(
        name="Live Update", default=True,
        description="Rebuild surfaces while their input curves are edited")
    show_tree: BoolProperty(name="Construction Tree", default=True,
                            description="Show what the active object is built from and what uses it")
    chord_tolerance: FloatProperty(
        name="Chord Tolerance", default=0.0005, min=0.00005, max=0.05, precision=4, update=_retess,
        description="Max chord deviation, relative to the object's size")
    edge_angle_tolerance: FloatProperty(
        name="Edge Angle", default=2.0, min=0.25, max=45.0, update=_retess,
        description="Angle tolerance along edges (degrees). Finer than the surface one so neighbouring "
                    "objects' meshes meet without visible slits along shared curved edges")
    max_edge_rel: FloatProperty(
        name="Max Edge", default=1.0 / 30.0, min=0.0, max=0.5, precision=3, update=_retess,
        description="Longest triangle edge, relative to the object size (0 = no limit). Prevents long "
                    "sliver triangles that shade as dark bands on ruled / revolved surfaces")
    angle_tolerance: FloatProperty(
        name="Angle Tolerance", default=5.0, min=2.0, max=45.0, update=_retess,
        description="Max angle between neighbouring facets (degrees). Lower is smoother highlights on curved "
                    "faces and more triangles there (flat faces stay a few triangles)")


class NprotoView(bpy.types.PropertyGroup):
    """Display toggles. They live on the window manager, which is not part of
    undo: flipping them neither adds an undo step nor closes the last
    operator's Adjust panel (e.g. turning CVs on while tuning a fillet)."""
    # analysis settings (here, not on the scene: changing them adds no undo step)
    zebra_direction: EnumProperty(update=_rd, name="Stripes", items=[("VERTICAL", "Vertical", ""), ("HORIZONTAL", "Horizontal", "")])
    zebra_stripes: FloatProperty(update=_rd, name="Density", default=12.0, min=1.0, soft_max=60.0)
    zebra_opacity: FloatProperty(update=_rd, name="Opacity", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                                 description="Opacity of the zebra / draft / curvature / matcap overlay over the "
                                             "surface's own shading (0 = only the shading)")
    zebra_angle: FloatProperty(update=_rd, name="Zebra Angle", default=1.0, min=0.25, max=15.0,
                               description="Angle tolerance of the zebra's own fine tessellation (degrees)")
    zebra_chord: FloatProperty(update=_rd, name="Zebra Chord", default=0.0005, min=0.00002, max=0.01, precision=5,
                               description="Chord tolerance of the zebra's tessellation, relative to the object size")
    draft_pull: EnumProperty(update=_rd, name="Pull", items=[("Z", "+Z", ""), ("-Z", "-Z", ""), ("X", "+X", ""), ("-X", "-X", ""),
                                                 ("Y", "+Y", ""), ("-Y", "-Y", ""), ("VIEW", "View", "Towards the viewer")])
    draft_angle: FloatProperty(update=_rd, name="Min Draft °", default=1.0, min=0.0, max=45.0,
                               description="Green: more draft than this; yellow: within +-this; red: undercut")
    curv_kind: EnumProperty(update=_rd, name="Curvature", items=[("GAUSS", "Gaussian", "k1 x k2"), ("MEAN", "Mean", "(k1 + k2) / 2"),
                                                     ("MAX", "Max", "k1"), ("MIN", "Min", "k2")], default="MEAN")
    curv_auto: BoolProperty(update=_rd, name="Auto Range", default=True)
    curv_min: FloatProperty(update=_rd, name="Min", default=-1.0)
    curv_max: FloatProperty(update=_rd, name="Max", default=1.0)
    curv_bands: FloatProperty(update=_rd, name="Contours", default=10.0, min=0.0, max=50.0, description="Contour lines (0 = off)")
    curv_angle: FloatProperty(update=_rd, name="Curvature Quality °", default=5.0, min=0.5, max=15.0)
    curv_chord: FloatProperty(update=_rd, name="Curvature Chord", default=0.002, min=0.0001, max=0.02, precision=4)
    matcap_style: EnumProperty(update=_rd, name="Matcap", items=[
        # fixed numbers: Blender stores the number, and 1 was the old Red style
        ("HIGHLIGHT", "Highlight", "Dark material with one bright highlight band (fixed to the view)", 0),
        ("LINES", "Gray Lines", "Grey material with thin reflection lines (fixed to the view)", 2),
        ("ENVIRONMENT", "Environment", "Paint with a clear-coat reflection of a light tunnel, a horizon or an HDRI. "
                                       "Fixed to the world: orbiting sweeps the reflections across the part", 3)])
    env_source: EnumProperty(update=_rd, name="Environment", items=_env_source_items,
                             description="What the surfaces reflect")
    env_rotation: FloatProperty(update=_rd, name="Rotation", default=0.0, subtype='ANGLE', soft_min=-3.14159265,
                                soft_max=3.14159265, description="Turn the environment about the world Z axis "
                                                                 "(sweeps the reflections across the part)")
    env_height: FloatProperty(update=_rd, name="Height", default=0.0, min=-0.9, max=0.9,
                              description="Where the horizon sits (-0.9 low .. 0.9 high)")
    env_lights: IntProperty(update=_rd, name="Lights", default=8, min=2, max=32,
                            description="Strip lights around the tunnel")
    env_paint: FloatVectorProperty(update=_rd, name="Paint", subtype='COLOR', size=3, min=0.0, max=1.0,
                                   default=(0.012, 0.012, 0.014), description="Paint colour under the clear coat")
    env_gloss: FloatProperty(update=_rd, name="Gloss", default=1.0, min=0.0, soft_max=3.0, max=10.0,
                             description="Strength of the clear-coat reflection (0 = matte paint)")
    env_roughness: FloatProperty(update=_rd, name="Roughness", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                                 description="Blurs the reflection: 0 mirror, 0.3 satin, 1 matte")
    env_metallic: FloatProperty(update=_rd, name="Metallic", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                                description="0 = paint under a clear coat; 1 = bare metal, reflecting in the paint "
                                            "colour (anodised aluminium, chrome with a light grey)")
    env_coat: BoolProperty(update=_rd, name="Clear Coat", default=False,
                           description="Glossy reflection of the environment over Draft Angle, Curvature Map "
                                       "and the Highlight / Gray Lines matcaps (fixed to the world)")
    matcap_pos: FloatProperty(update=_rd, name="Position", default=0.0, min=-1.0, max=1.0,
                              description="Where the highlight band sits (view-space, -1..1)")
    matcap_width: FloatProperty(update=_rd, name="Width", default=0.06, min=0.005, max=0.5)
    matcap_lines: FloatProperty(update=_rd, name="Lines", default=6.0, min=1.0, soft_max=40.0)
    matcap_line_px: FloatProperty(update=_rd, name="Line Px", default=0.6, min=0.0, max=6.0, description="Line thickness in pixels")
    comb_scale: FloatProperty(update=_rd, name="Scale", default=1.0, min=0.01, soft_max=10.0,
                              description="Comb length: at 1 the longest hair is a fifth of the curve's size")
    comb_samples: IntProperty(update=_rd, name="Samples", default=200, min=8, max=2000,
                              description="Points per curve")
    diag_selected_only: BoolProperty(update=_rd, name="Selected Only", default=True,
                                     description="Zebra / comb on the selected objects only (else on all visible)")
    sel_mode: EnumProperty(name="Select", default="BODY", update=lambda s, c: _subsel_mode(s, c),
                           items=[("BODY", "Body", "Select whole bodies (Blender's object selection)", 'MESH_CUBE', 0),
                                  ("FACE", "Face", "Pick faces of NURBS bodies: click, Shift+click adds or removes. "
                                                   "CVs / Hulls then shows only the picked faces' nets", 'FACESEL', 1),
                                  ("EDGE", "Edge", "Pick edges of NURBS bodies: click, Shift+click adds or removes. "
                                                   "Edge Fillet and Continuity Check take the picked edges", 'EDGESEL', 2)])
    show_edges: BoolProperty(name="Edges", default=True, update=lambda s, c: _redraw_views(c))
    show_cvs: BoolProperty(name="CVs / Hulls", default=False, update=lambda s, c: _redraw_views(c))
    show_cont_labels: BoolProperty(name="Feature Labels", default=True,
                                   description="The continuity each feature was BUILT with: Free / G0 / G1 / G2 labels on the "
                                               "edges of selected Square, Loft, Loft Rails, Blend Curve, Patch and Surface Align "
                                               "features. Click a label to change the setting (Ctrl+click steps back; the feature "
                                               "rebuilds); a red ring means the result misses it. To MEASURE any edge, use "
                                               "Analyze > Continuity Check", update=lambda s, c: _redraw_views(c))
    cont_show_checks: BoolProperty(name="Show Checks", default=True, update=lambda s, c: _redraw_views(c),
                                   description="Show the labels and combs of the edges picked with Continuity Check")
    cont_level: EnumProperty(name="Show", update=lambda s, c: _redraw_views(c), default='G2',
                             items=[('G0', "G0", "Position: the gap between the two sides (comb = gap, magnified)"),
                                    ('G1', "G1", "Tangency: the angle between the two sides' normals (comb = angle)"),
                                    ('G2', "G2", "Curvature: the difference of the two sides' curvature across the edge "
                                                 "(comb = difference, magnified; Curvature Combs: each side's own too)")])
    cont_scale: FloatProperty(name="Scale", default=1.0, min=0.02, soft_max=10.0, max=50.0,
                              update=lambda s, c: _redraw_views(c), description="Comb length: at 1 the longest hair is a fifth of the edge's length (+ / - in the tool)")
    cont_kcombs: BoolProperty(name="Curvature Combs", default=False, update=lambda s, c: _redraw_views(c),
                              description="G2: also draw each side's own curvature across the edge (white / magenta); "
                                          "where the edge is G2 the two lie on each other")
    cont_samples: IntProperty(name="Samples", default=160, min=16, max=2000, update=lambda s, c: _redraw_views(c),
                              description="Points measured along each checked edge. A dense surface needs about 4 per CV "
                                          "for its true comb (the panel says how many). More: smoother, slower")
    cont_gap_tol: FloatProperty(name="Gap", default=1e-5, min=0.0, precision=6, unit='LENGTH',
                                update=lambda s, c: _redraw_views(c),
                                description="G0 holds while the two sides are no farther apart than this")
    cont_angle_tol: FloatProperty(name="Angle °", default=0.1, min=0.0, soft_max=5.0, precision=3,
                                  update=lambda s, c: _redraw_views(c),
                                  description="G1 holds while the normals of the two sides differ by no more than this "
                                              "(also used by the feature labels' red ring)")
    cont_curv_tol: FloatProperty(name="Curvature %", default=5.0, min=0.0, soft_max=50.0,
                                 update=lambda s, c: _redraw_views(c),
                                 description="G2 holds while the curvature across the edge differs by no more than this, "
                                             "in percent of the largest along the edge (also used by the feature labels)")
    show_zebra: BoolProperty(name="Zebra", default=False, update=lambda s, c: _one_map(s, "show_zebra"),
                             description="Reflection stripes on NURBS surfaces")
    show_draft: BoolProperty(name="Draft Angle", default=False, update=lambda s, c: _one_map(s, "show_draft"),
                             description="Colour surfaces by draft angle against a pull direction")
    show_curvature: BoolProperty(name="Curvature Map", default=False, update=lambda s, c: _one_map(s, "show_curvature"),
                                 description="Colour surfaces by exact curvature")
    show_matcap: BoolProperty(name="Matcap", default=False, update=lambda s, c: _one_map(s, "show_matcap"),
                              description="Highlight, grey-lines or environment shading on NURBS surfaces (exact normals)")
    show_iso_combs: BoolProperty(name="Surface Combs", default=False, update=lambda s, c: _redraw_views(c),
                                 description="Curvature combs on iso-curves of the NURBS surfaces (Alias Curvature U / V)")
    iso_along_u: BoolProperty(name="Along U", default=True, update=lambda s, c: _redraw_views(c),
                              description="Combs on the iso-curves running along U")
    iso_along_v: BoolProperty(name="Along V", default=True, update=lambda s, c: _redraw_views(c),
                              description="Combs on the iso-curves running along V")
    iso_count_u: IntProperty(name="Number U", default=8, min=1, max=100, update=lambda s, c: _redraw_views(c),
                             description="How many iso-curves along U, evenly through the face")
    iso_count_v: IntProperty(name="Number V", default=6, min=1, max=100, update=lambda s, c: _redraw_views(c),
                             description="How many iso-curves along V, evenly through the face")
    iso_samples: IntProperty(name="Samples", default=100, min=8, max=2000, update=lambda s, c: _redraw_views(c),
                             description="Points per iso-curve")
    iso_scale: FloatProperty(name="Scale", default=1.0, min=0.0, soft_max=10.0, update=lambda s, c: _redraw_views(c),
                             description="1: the longest combs are about a fifth of the object's size")
    show_comb: BoolProperty(name="Curvature Comb", default=False, description="Curvature comb on curves", update=lambda s, c: _redraw_views(c))


def view(context=None):
    return getattr((context or bpy.context).window_manager, "nproto_view", None)


def coat_applies(v):
    """The Clear Coat goes over Draft Angle, Curvature Map and the view-fixed matcaps."""
    return v.show_draft or v.show_curvature or (v.show_matcap and v.matcap_style != 'ENVIRONMENT')


def _draw_env(col, v, material):
    """The environment's controls; `material`: also the paint's (Environment
    matcap), not for the Clear Coat over another overlay."""
    col.prop(v, "env_source", text="")
    row = col.row(align=True)
    if v.env_source == 'HORIZON':
        row.prop(v, "env_height")
    else:
        row.prop(v, "env_rotation")
        if v.env_source == 'TUNNEL':
            row.prop(v, "env_lights")
    row = col.row(align=True)
    if material:
        row.prop(v, "env_paint", text="")
    row.prop(v, "env_gloss")
    row = col.row(align=True)
    row.prop(v, "env_roughness", slider=True)
    if material:
        row.prop(v, "env_metallic", slider=True)


def draw_analyze(col, context):
    """Zebra / Draft / Curvature / Comb / Matcap toggles and their settings
    (sidebar and the viewport header's Analyze popover)."""
    v = context.window_manager.nproto_view
    row = col.row(align=True)
    row.prop(v, "show_zebra", toggle=True, icon='MOD_WAVE')
    row.prop(v, "show_draft", toggle=True, icon='MOD_SIMPLEDEFORM')
    row = col.row(align=True)
    row.prop(v, "show_curvature", toggle=True, icon='COLOR')
    row.prop(v, "show_comb", toggle=True, icon='IPO_ELASTIC')
    row = col.row(align=True)
    row.prop(v, "show_iso_combs", toggle=True, icon='MOD_LATTICE')
    row.prop(v, "show_matcap", toggle=True, icon='MATSPHERE')
    if v.show_iso_combs:
        row = col.row(align=True)
        row.prop(v, "iso_along_u", toggle=True)
        row.prop(v, "iso_count_u", text="")
        row.prop(v, "iso_along_v", toggle=True)
        row.prop(v, "iso_count_v", text="")
        row = col.row(align=True)
        row.prop(v, "iso_scale")
        row.prop(v, "iso_samples")
    if v.show_matcap:
        col.row(align=True).prop(v, "matcap_style", expand=True)
        if v.matcap_style == 'ENVIRONMENT':
            _draw_env(col, v, material=True)
        else:
            col.row(align=True).prop(v, "zebra_direction", expand=True)
            row = col.row(align=True)
            if v.matcap_style == 'LINES':
                row.prop(v, "matcap_lines")
                row.prop(v, "matcap_line_px")
            else:
                row.prop(v, "matcap_pos")
                row.prop(v, "matcap_width")
    if v.show_draft:
        row = col.row(align=True)
        row.prop(v, "draft_pull", text="Pull")
        row.prop(v, "draft_angle")
        col.label(text="green: draft  ·  yellow: within ±  ·  red: undercut", icon='INFO')
    if v.show_curvature:
        col.row(align=True).prop(v, "curv_kind", expand=True)
        row = col.row(align=True)
        row.prop(v, "curv_auto")
        row.prop(v, "curv_bands")
        if v.curv_auto:
            from . import diagnostics as _dg
            rng = _dg.LAST_RANGE
            if rng:
                col.label(text=f"blue {rng[0]:.4g}  ->  red {rng[1]:.4g}  (1/length)", icon='COLOR')
        else:
            row = col.row(align=True)
            row.prop(v, "curv_min")
            row.prop(v, "curv_max")
        row = col.row(align=True)
        row.prop(v, "curv_angle", text="Quality °")
        row.prop(v, "curv_chord", text="Chord")
    if v.show_zebra:
        row = col.row(align=True)
        row.prop(v, "zebra_direction", expand=True)
        col.prop(v, "zebra_stripes")
        row = col.row(align=True)
        row.prop(v, "zebra_angle", text="Quality °")
        row.prop(v, "zebra_chord", text="Chord")
    if v.show_comb:
        row = col.row(align=True)
        row.prop(v, "comb_scale")
        row.prop(v, "comb_samples")
    if coat_applies(v):
        col.prop(v, "env_coat", toggle=True, icon='SHADING_RENDERED')
        if v.env_coat:
            _draw_env(col, v, material=False)
    if v.show_zebra or v.show_draft or v.show_curvature or v.show_matcap:
        row = col.row(align=True)
        row.prop(v, "zebra_opacity", slider=True)
        sp = context.space_data
        if sp is not None and sp.type == 'VIEW_3D':
            row.prop(sp.shading, "show_xray", text="See-Through", toggle=True, icon='XRAY')
    if v.show_zebra or v.show_comb or v.show_iso_combs or v.show_draft or v.show_curvature or v.show_matcap:
        col.prop(v, "diag_selected_only")


class VIEW3D_PT_nproto_analyze(bpy.types.Panel):
    """Analyze overlays, reachable from the viewport header at any time -- also
    while a tool is picking or its Adjust Last Operation panel is open"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "NURBS Analyze"
    bl_ui_units_x = 12

    def draw(self, context):
        from . import contcheck, sections
        self.layout.label(text="NURBS Analyze")
        col = self.layout.column(align=True)
        contcheck.draw_panel(col, context)
        col.separator()
        sections.draw_panel(col, context)
        col.separator()
        draw_analyze(col, context)


def _subsel_mode(view, context):
    from . import subsel
    subsel.mode_changed(view, context)


def header_analyze(self, context):
    v = getattr(context.window_manager, "nproto_view", None)
    if v is None:
        return
    from . import subsel
    subsel.draw_mode_buttons(self.layout, context)
    on = [n for n, k in (("Zebra", "show_zebra"), ("Draft", "show_draft"), ("Curvature", "show_curvature"),
                         ("Matcap", "show_matcap"), ("Comb", "show_comb"), ("Surface Combs", "show_iso_combs"))
          if getattr(v, k)]
    if v.env_coat and coat_applies(v):
        on.append("Coat")
    self.layout.popover("VIEW3D_PT_nproto_analyze", text=" · ".join(on) if on else "", icon='MOD_WAVE')


class _SidePanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NURBS Proto"


def _cage(context):
    from . import cvs as _cvs
    act = context.active_object
    if act is None:
        return None
    return act if _cvs.is_cage(act) else _cvs.cage_for(act)


class VIEW3D_PT_nproto_active(_SidePanel, bpy.types.Panel):
    """The active object: CV editing, its feature settings and history.
    Shown only when there is something to show (the tool panels follow)"""
    bl_label = "Active Object"
    bl_order = 0

    @classmethod
    def poll(cls, context):
        if load_kernel() is None or _cage(context) is not None:
            return True
        obj = context.view_layer.objects.active
        if obj is None:
            return False
        from . import feature_ui as _fui
        return store.is_nurbs(obj) or _fui.shows_feature(obj, context)

    def draw(self, context):
        lay = self.layout
        if load_kernel() is None:
            from . import kernel
            lay.label(text="Kernel not loaded", icon='ERROR')
            lay.label(text=kernel.error or "")
            return
        from . import cvs as _cvs
        cage = _cage(context)
        if cage is not None:
            box = lay.box()
            box.alert = True
            box.label(text=f"Editing CVs of {cage[_cvs.CAGE_OF]}", icon='EDITMODE_HLT')
            box.label(text="G / R / S, proportional editing, snapping...")
            src = bpy.data.objects.get(cage[_cvs.CAGE_OF])
            if src is not None and features.has_history(src):
                box.label(text="Moving a CV removes its history (Cancel keeps it)", icon='INFO')
            row = box.row(align=True)
            row.scale_y = 1.3
            row.operator("nproto.cvs_done", icon='CHECKMARK')
            row.operator("nproto.cvs_cancel", icon='X')

        from . import feature_ui as _fui
        _fui.draw_feature(lay, context)

        obj = context.view_layer.objects.active
        if store.is_nurbs(obj):
            box = lay.box()
            brep = store.get_brep(obj)
            box.label(text=f"{obj.name}: {brep.face_count()} face(s), {brep.edge_count()} edge(s)")
            h = features.history(obj)
            if h:
                op = h["op"].replace("prim_", "").replace("_", " ").title()
                box.label(text=f"History: {op}" + (" from " + ", ".join(o.name for o, _ in h["inputs"]) if h["inputs"] else ""))
            elif features.PROP_HIST in obj:
                box.label(text="History: broken (input deleted)", icon='ERROR')
            from . import subsel
            picked = subsel.summary(obj)
            if picked:
                box.label(text=picked, icon='FACESEL' if subsel.get(obj)[0] == "FACE" else 'EDGESEL')


class VIEW3D_PT_nproto_analyze_side(_SidePanel, bpy.types.Panel):
    """Continuity check and the analysis overlays (also in the viewport header)"""
    bl_label = "Analyze"
    bl_order = 5
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return load_kernel() is not None

    def draw(self, context):
        from . import contcheck, sections
        col = self.layout.column(align=True)
        contcheck.draw_panel(col, context)
        col.separator()
        sections.draw_panel(col, context)
        col.separator()
        draw_analyze(col, context)


class VIEW3D_PT_nproto_settings(_SidePanel, bpy.types.Panel):
    """History and display settings"""
    bl_label = "Settings"
    bl_order = 6
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return load_kernel() is not None

    def draw(self, context):
        lay = self.layout
        s = context.scene.nproto_settings
        v = context.window_manager.nproto_view
        col = lay.column(align=True)
        col.label(text="History")
        col.prop(s, "construction_history")
        sub = col.row()
        sub.enabled = s.construction_history
        sub.prop(s, "live_update")
        draw_pause(col, context)
        row = col.row(align=True)
        row.operator("nproto.delete_history", text="Delete History").scope = 'SELECTED'
        row.operator("nproto.delete_history", text="All").scope = 'ALL'
        col.operator("nproto.collect_inputs", icon='OUTLINER_COLLECTION')
        col.operator("nproto.rebuild", icon='FILE_REFRESH')

        col = lay.column(align=True)
        col.label(text="Display")
        row = col.row(align=True)
        row.prop(v, "show_edges", toggle=True)
        row.prop(v, "show_cvs", toggle=True)
        row.prop(v, "show_cont_labels", text="Feature Labels", toggle=True)
        col.prop(s, "chord_tolerance")
        col.prop(s, "angle_tolerance")
        col.prop(s, "edge_angle_tolerance")
        col.prop(s, "max_edge_rel")


classes = (NPROTO_OT_rebuild_waiting, NprotoSettings, NprotoView, VIEW3D_PT_nproto_active, VIEW3D_PT_nproto_analyze,
           VIEW3D_PT_nproto_analyze_side, VIEW3D_PT_nproto_settings)


def register():
    bpy.types.Scene.nproto_settings = PointerProperty(type=NprotoSettings)
    bpy.types.WindowManager.nproto_view = PointerProperty(type=NprotoView)
    bpy.types.VIEW3D_HT_header.append(header_analyze)
    from . import subsel
    bpy.types.VIEW3D_HT_tool_header.prepend(subsel.draw_tool_header)


def unregister():
    from . import subsel
    bpy.types.VIEW3D_HT_tool_header.remove(subsel.draw_tool_header)
    bpy.types.VIEW3D_HT_header.remove(header_analyze)
    del bpy.types.Scene.nproto_settings
    del bpy.types.WindowManager.nproto_view
