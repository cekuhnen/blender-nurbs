"""Align Parts with simulated input in a real window (hover, clicks, keys,
hold A, the toolbar tool):
    blender -b --factory-startup --python tests/test_align_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~30 s), and reads back what
that run found. Simulated input sends no INBETWEEN_MOUSEMOVE: drags on the
navigation gizmo still need a real mouse."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--align-gui" in sys.argv


# ======================================================================================
# the background run: start the GUI run, report its findings
# ======================================================================================
def background():
    tmp = tempfile.mkdtemp(prefix="nproto_align_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--align-gui", out]
    try:
        subprocess.run(cmd, timeout=240, capture_output=True)
    except subprocess.TimeoutExpired:
        print("FAIL the GUI run did not finish in 240 s")
        os._exit(1)
    if not os.path.isfile(out):
        print("FAIL the GUI run wrote no results")
        os._exit(1)
    res = json.load(open(out))
    ok = True
    for name, (good, extra) in res["checks"].items():
        print(("PASS " if good else "FAIL ") + name, extra if not good else "")
        ok &= bool(good)
    if res.get("error"):
        print("FAIL the GUI run raised\n" + res["error"])
        ok = False
    print("ALL PASS" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


# ======================================================================================
# the GUI run
# ======================================================================================
def gui():
    import traceback
    import numpy as np
    from mathutils import Matrix, Vector
    from bpy_extras import view3d_utils
    sys.path.insert(0, ROOT)
    import blender_nurbs
    blender_nurbs.register()
    from blender_nurbs import aligntool, store

    out = sys.argv[sys.argv.index("--align-gui") + 1]
    checks = {}
    res = {"checks": checks}

    def check(name, cond, extra=""):
        checks[name] = (bool(cond), str(extra))

    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    over = lambda: ctx.temp_override(window=win, area=area, region=region)

    def wpx(P, d=(0, 0)):
        q = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(P))
        return int(round(region.x + q[0] + d[0])), int(round(region.y + q[1] + d[1]))

    at = [0, 0]

    def move(p):
        """The mouse to window pixel p, in a few steps (as a hand would)."""
        for k in (3, 2, 1, 0):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0] + k, y=p[1] + k)
        at[:] = p

    def key(t, uni=None, shift=False):
        kw = {"unicode": uni} if uni else {}
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1], shift=shift, **kw)
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1], shift=shift)

    def click():
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1])
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])

    modal = lambda: [op.bl_idname for op in win.modal_operators]
    running = lambda: "NPROTO_OT_align_parts" in modal()
    Mw = lambda o: np.array(o.matrix_world, float)
    close = lambda a, b, tol=1e-5: bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))

    def top_view(loc, dist):
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'ORTHO'
        r3.view_rotation = Matrix.Identity(3).to_quaternion()
        r3.view_location = loc
        r3.view_distance = dist

    def deselect():
        for o in ctx.view_layer.objects:
            o.select_set(False)

    def steps():
        # -- a NURBS box and a peg ----------------------------------------------------
        with over():
            bpy.ops.nproto.add_primitive(kind='BOX', width=2.0, depth=2.0, height=1.0)
            box = ctx.view_layer.objects.active
            box.name = "Box"
            bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
            peg = ctx.view_layer.objects.active
            peg.name = "Peg"
        M0 = Matrix.Translation((4.0, 3.0, 0.0))
        peg.matrix_world = M0
        deselect()
        with over():
            bpy.ops.ed.undo_push(message="setup")
        top_view((2.0, 1.5, 0.0), 12.0)
        yield 0.5
        zt = max(f.centroid()[2] for f in store.get_brep(peg).faces())
        zb = max(f.centroid()[2] for f in store.get_brep(box).faces())
        pa, pb = wpx((4.0, 3.0, zt), (1, 1)), wpx((0.0, 0.0, zb), (2, 2))
        top_w = lambda o: (Mw(o) @ np.array([0, 0, zt, 1.0]))[:3]

        # -- tap A: Blender's Select All ------------------------------------------------
        move(pa)
        yield 0.2
        win.event_simulate('A', 'PRESS', x=at[0], y=at[1])
        yield 0.05
        win.event_simulate('A', 'RELEASE', x=at[0], y=at[1])
        yield 0.3
        check("tap A: Select All, no Align Parts", sorted(o.name for o in ctx.selected_objects) == ["Box", "Peg"]
              and not running(), (sorted(o.name for o in ctx.selected_objects), modal()))
        deselect()
        yield 0.2

        # -- hold A: Align Parts; ring onto the centre, a gap of 1, flipped ----------------
        win.event_simulate('A', 'PRESS', x=at[0], y=at[1])
        yield 0.6
        check("hold A: Align Parts starts", running(), modal())
        win.event_simulate('A', 'RELEASE', x=at[0], y=at[1])
        yield 0.1
        move(pa)
        yield 0.1
        click()
        yield 0.2
        move(pb)
        yield 0.2
        check("hovering the target: nothing moves yet", close(Mw(peg), M0))
        click()
        yield 0.1
        key('ONE', '1')
        key('F')
        yield 0.2
        check("refine stage: still running after the target click", running(), modal())
        key('RET')
        yield 0.4
        check("hold A run: done on Enter", not running(), modal())
        check("hold A run: flipped, top 1 above the box's centre", close(top_w(peg), (0, 0, zb + 1.0))
              and close(Mw(peg)[:3, 2], (0, 0, 1)), (top_w(peg), Mw(peg)[:3, 2]))
        check("hold A run: the peg selected", [o.name for o in ctx.selected_objects] == ["Peg"])
        with over():
            bpy.ops.ed.undo()
        yield 0.4
        peg = bpy.data.objects["Peg"]
        check("undo: one step puts the peg back", close(Mw(peg), M0), Mw(peg)[:3, 3])

        # -- Esc cancels -------------------------------------------------------------------
        with over():
            bpy.ops.nproto.align_parts('INVOKE_DEFAULT')
        yield 0.2
        move(pa)
        yield 0.1
        click()
        yield 0.1
        move(pb)
        yield 0.1
        key('ESC')
        yield 0.3
        check("Esc: cancelled, nothing moved", not running() and close(Mw(peg), M0))

        # -- the toolbar tool: its click is the first element; copy, spin --------------------
        with over():
            bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
        yield 0.3            # (drawn once with select box: the tool to come back)
        with over():
            bpy.ops.wm.tool_set_by_id(name=aligntool.TOOL_ID)
        yield 0.3
        n0 = len(bpy.data.objects)
        move(pa)
        yield 0.1
        click()
        yield 0.3
        check("toolbar tool: its click picks the peg's ring and runs Align Parts", running(), modal())
        move(pb)
        yield 0.1
        click()
        yield 0.1
        key('D')
        key('R', shift=True)
        yield 0.2
        key('RET')
        yield 0.5
        new = [o for o in bpy.data.objects if o.name not in ("Box", "Peg")]
        check("toolbar run: the original stays, a copy on the box, upside down", close(Mw(peg), M0)
              and len(bpy.data.objects) == n0 + 1 and len(new) == 1 and close(top_w(new[0]), (0, 0, zb))
              and close(Mw(new[0])[:3, 2], (0, 0, -1)), [(o.name, top_w(o)) for o in new])
        if new:
            x = Mw(new[0])[:3, 0]
            check("toolbar run: Shift+R spun it 15 degrees about the axis",
                  abs(abs(math.degrees(math.atan2(x[1], x[0]))) - 15.0) < 1e-3 or
                  abs(abs(math.degrees(math.atan2(x[1], x[0]))) - 165.0) < 1e-3, x)
        tool = ctx.workspace.tools.from_space_view3d_mode('OBJECT', create=False)
        check("toolbar run: the tool before it comes back", tool is not None and tool.idname == "builtin.select_box",
              tool.idname if tool else None)

        # -- meshes: face on face, a spin, then a second pair of edges undoes it ---------------
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        with over():
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
            base = ctx.view_layer.objects.active
            base.name = "Base"
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(5.0, 0.0, 0.0), rotation=(0, 0, math.radians(30)))
            blk = ctx.view_layer.objects.active
            blk.name = "Block"
        deselect()
        top_view((2.5, 0.0, 0.0), 14.0)
        yield 0.5
        R30 = np.array(Matrix.Rotation(math.radians(30), 3, 'Z'))
        on_blk = lambda u, v: tuple(np.array([5.0, 0.0, 1.0]) + R30 @ np.array([u, v, 0.0]))
        with over():
            bpy.ops.nproto.align_parts('INVOKE_DEFAULT')
        yield 0.2
        move(wpx(on_blk(-0.5, 0.4)))
        yield 0.1
        click()
        yield 0.1
        move(wpx((-0.4, 0.3, 1.0)))
        yield 0.1
        click()
        yield 0.1
        key('R', shift=True)
        key('R', shift=True)            # 30 degrees off the base's edges
        yield 0.1
        move(wpx(on_blk(1.0, 0.4)))     # the block's edge (in its place before the move)
        yield 0.1
        click()
        yield 0.1
        move(wpx((-0.4, -1.0, 1.0)))    # the base's front edge
        yield 0.1
        click()
        yield 0.1
        key('RET')
        yield 0.4
        R = Mw(blk)[:3, :3]
        square = all(min(abs(abs(float(R[:, k] @ e)) - 1.0) for e in np.eye(3)) < 1e-5 for k in range(3))
        check("second pair: the edges line up again (the spin set by them)", square, np.round(R, 4).tolist())
        check("second pair: still flush on the base, upside down", abs(Mw(blk)[2, 3] - 2.0) < 1e-5
              and close(R[:, 2], (0, 0, -1)), Mw(blk)[:3, 3])

    g = steps()

    def tick():
        try:
            return next(g)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        json.dump(res, open(out, "w"), indent=1, default=str)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
