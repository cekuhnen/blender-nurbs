"""Surface / curve diagnostics drawn in the viewport overlay.

* Curvature comb on curves (Blender NURBS / Bezier / poly curves, exact via the
  kernel): a "hair" at every sample, length = curvature x scale, pointing away
  from the centre of curvature, plus the envelope through the hair tips.
  Inflections show as the comb switching sides, curvature jumps (G1 joins) as
  steps in the envelope.
* Zebra stripes on NURBS surfaces: reflection stripes computed per pixel by a
  small GPU shader (any shading mode, materials untouched). Stripes run
  vertically or horizontally on screen. G0 -> stripes break, G1 -> stripes
  kink, G2 -> stripes flow smoothly across an edge.
"""

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import geom, store

COL_COMB = (0.1, 0.85, 1.0, 0.85)
COL_ENVELOPE = (1.0, 0.35, 0.2, 1.0)


# --------------------------------------------------------------------------
# Curvature comb
# --------------------------------------------------------------------------

def curvature_samples(kcurve, n):
    """(points, curvature vectors) along a kernel curve: kappa = |k|, k points to
    the centre of curvature. Uses the exact 1st / 2nd derivatives."""
    from .kernel import load
    sm = load()
    t0, t1 = kcurve.parameter_range()
    P, Kv = [], []
    for t in np.linspace(t0, t1, n):
        d = sm.evaluate_curve(kcurve, float(t))
        p = np.array(d["point"], float)
        d1 = np.array(d["tangent"], float)
        d2 = np.array(d["curvature"], float)
        s = float(d1 @ d1)
        k = np.cross(np.cross(d1, d2), d1) / (s * s) if s > 1e-30 else np.zeros(3)
        P.append(p)
        Kv.append(k)
    return np.array(P), np.array(Kv)


_comb_cache = {}


def _curve_key(obj):
    cu = obj.data
    parts = [tuple(obj.matrix_world.col[i][:] for i in range(4))]
    for sp in cu.splines:
        if sp.type == 'BEZIER':
            a = np.zeros(len(sp.bezier_points) * 9)
            co = np.zeros(len(sp.bezier_points) * 3)
            sp.bezier_points.foreach_get("co", co)
            hl = np.zeros_like(co); hr = np.zeros_like(co)
            sp.bezier_points.foreach_get("handle_left", hl)
            sp.bezier_points.foreach_get("handle_right", hr)
            a = np.concatenate([co, hl, hr])
        else:
            a = np.zeros(len(sp.points) * 4)
            sp.points.foreach_get("co", a)
        parts.append((sp.type, sp.order_u, sp.use_endpoint_u, sp.use_bezier_u, sp.use_cyclic_u,
                      hash(a.round(9).tobytes())))
    return hash(tuple(parts))


def comb_data(obj, samples):
    """Cached comb samples for every spline of a curve object (world space)."""
    key = (_curve_key(obj), samples)
    hit = _comb_cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    out = []
    M = geom.world_matrix(obj)
    for sp in obj.data.splines:
        try:
            kc = geom.spline_to_kernel_curve(sp, M)
            out.append(curvature_samples(kc, samples))
        except Exception:
            continue
    _comb_cache[obj.session_uid] = (key, out)
    return out


def draw_combs(ctx, draw_lines):
    s = ctx.scene.nproto_settings
    v = ctx.window_manager.nproto_view
    if not v.show_comb:
        return
    objs = [o for o in (ctx.selected_objects if v.diag_selected_only else ctx.visible_objects) if o.type == 'CURVE']
    if not objs:
        return
    gpu.state.depth_test_set('NONE')
    for obj in objs:
        try:
            data = comb_data(obj, int(v.comb_samples))
        except Exception:
            continue
        # scale: comb_scale x (object size / max curvature) keeps the comb readable
        for P, Kv in data:
            kap = np.linalg.norm(Kv, axis=1)
            size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
            ref = float(kap.max()) or 1.0
            L = v.comb_scale * 0.3 * size / ref
            tips = P - L * Kv                     # hairs point away from the centre of curvature
            hairs = []
            for a, b in zip(P, tips):
                hairs += [Vector(a), Vector(b)]
            env = []
            for a, b in zip(tips[:-1], tips[1:]):
                env += [Vector(a), Vector(b)]
            draw_lines(hairs, COL_COMB, 1.0)
            draw_lines(env, COL_ENVELOPE, 2.0)
    gpu.state.depth_test_set('LESS_EQUAL')


# --------------------------------------------------------------------------
# Zebra
# --------------------------------------------------------------------------

_shader = None
_zebra_cache = {}
_curv_cache = {}

_VS = """
void main()
{
    vec4 wp = modelMatrix * vec4(pos, 1.0);
    vP = wp.xyz;
    vN = normalize(mat3(normalMatrix) * nor);
    vV = val;
    vec3 towards = normalize(eye - wp.xyz);
    /* stay on top of the shaded mesh: its coarse triangles lie in front of
       the fine tessellation in hollows, so pull towards the eye a little */
    gl_Position = viewProjectionMatrix * vec4(wp.xyz + towards * bias, 1.0);
    gl_Position.z -= 2e-5 * gl_Position.w;
}
"""

_FS = """
vec3 ramp(float t)          /* blue - cyan - green - yellow - red */
{
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.10, 0.25, 1.00), c1 = vec3(0.00, 0.85, 1.00), c2 = vec3(0.10, 0.90, 0.20);
    vec3 c3 = vec3(1.00, 0.90, 0.10), c4 = vec3(1.00, 0.15, 0.10);
    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
}

vec4 lin(vec4 c) { return vec4(pow(max(c.rgb, vec3(0.0)), vec3(2.2)), c.a); }   /* sRGB-authored -> linear */

/* ---- Environment matcap: what the surface reflects (see environments.py) ---- */
vec3 env_frame(vec3 d)          /* world direction -> environment frame (turned about Z) */
{
    float c = cos(env_rot), s = sin(env_rot);
    return vec3(c * d.x + s * d.y, -s * d.x + c * d.y, d.z);
}

vec2 equirect(vec3 d)           /* v = 1 straight up */
{
    return vec2(0.5 - atan(d.y, d.x) / 6.2831853, 0.5 + asin(clamp(d.z, -1.0, 1.0)) / 3.1415927);
}

float pulse_avg(float x, float duty, float w)    /* mean over [x - w, x + w] of 1 on [k, k + duty), 0 elsewhere */
{
    w = max(w, 1e-5);
    float lo = x - w, hi = x + w;
    return ((floor(hi) * duty + min(fract(hi), duty)) - (floor(lo) * duty + min(fract(lo), duty))) / (2.0 * w);
}

/* blur: the angle (radians) the reflection is averaged over: one pixel, widened by the roughness */
vec3 env_tunnel(vec3 d, float blur)  /* strip lights around the X axis, one straight up */
{
    float s = length(d.yz);
    float f = (atan(d.z, d.y) - 1.5707963) / 6.2831853 * env_lights;   /* strip centres at whole numbers */
    float w = 0.5 * blur / max(s, 1e-3) * env_lights / 6.2831853;
    float strip = pulse_avg(f + 0.125, 0.25, w);                        /* box filter: exact at any blur */
    float bright = 0.6 + 0.4 * (0.5 + 0.5 * d.z / max(s, 1e-4));
    float opening = clamp((s - 0.15) / 0.2, 0.0, 1.0);
    return vec3(0.02 + strip * bright * opening);
}

vec3 env_horizon(vec3 d, float blur)
{
    float h = d.z - env_height;
    float e = clamp(h / max(1.0 - env_height, 1e-3), 0.0, 1.0);
    float g = clamp(-h / max(1.0 + env_height, 1e-3), 0.0, 1.0);
    vec3 sky = mix(vec3(1.0), vec3(0.35, 0.45, 0.65), e);
    vec3 ground = mix(vec3(0.03), vec3(0.01), g);
    return mix(ground, sky, smoothstep(-blur, blur, h));
}

vec3 env_hdri(vec3 d, float blur)    /* the 1024 / 256 / 64 wide maps, then the irradiance (fully rough) */
{
    vec2 uv = equirect(d);
    float lod = log2(max(blur * env_texels / 6.2831853, 1.0));
    if (lod < 2.0) return mix(textureLod(env_hi, uv, 0.0).rgb, textureLod(env_mid, uv, 0.0).rgb, lod * 0.5);
    if (lod < 4.0) return mix(textureLod(env_mid, uv, 0.0).rgb, textureLod(env_low, uv, 0.0).rgb, lod * 0.5 - 1.0);
    return mix(textureLod(env_low, uv, 0.0).rgb, textureLod(env_irr, uv, 0.0).rgb, clamp(lod * 0.5 - 2.0, 0.0, 1.0));
}

vec3 env_radiance(vec3 r, float blur)   /* what a surface reflects in world direction r */
{
    vec3 d = env_frame(r);
    return (style == 0) ? env_tunnel(d, blur) : ((style == 1) ? env_horizon(d, blur) : env_hdri(d, blur));
}

float env_blur(vec3 r)
{
    return length(vec2(length(fwidth(r)), env_rough * env_rough));
}

vec3 tonemap(vec3 x)            /* filmic curve: highlights roll off instead of clipping */
{
    x = max(x, vec3(0.0));
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec4 finish(vec3 col, vec3 n, vec3 v)    /* an overlay's sRGB colour -> linear, with the clear coat on top */
{
    vec3 base = pow(max(col, vec3(0.0)), vec3(2.2));
    if (coat_on == 1) {
        if (dot(n, v) > 0.0) n = -n;
        vec3 r = reflect(v, n);
        float F = 0.04 + 0.96 * pow(1.0 - clamp(-dot(n, v), 0.0, 1.0), 5.0);
        base = base * (1.0 - min(F * gloss, 1.0)) + tonemap(env_gain * gloss * F * env_radiance(r, env_blur(r)));
    }
    return vec4(base, opacity);
}

void main()
{
    vec3 n = normalize(vN);
    vec3 v = normalize(vP - eye);
    float shade = 0.55 + 0.45 * abs(dot(n, v));      /* keep some form in the flat colours */
    if (mode == 1) {                                 /* draft angle */
        float d = degrees(asin(clamp(dot(n, pull), -1.0, 1.0)));
        vec3 col = (d > thr) ? vec3(0.15, 0.80, 0.25) : ((d < -thr) ? vec3(0.90, 0.15, 0.12) : vec3(1.0, 0.85, 0.10));
        if (abs(abs(d) - thr) < 0.15) col *= 0.35;   /* thin line at the threshold */
        FragColor = finish(col * shade, n, v);
        return;
    }
    if (mode == 2) {                                 /* curvature map */
        float t = (vV - vmin) / max(vmax - vmin, 1e-12);
        vec3 col = ramp(t);
        if (fract(t * bands) < 0.04 && bands > 0.5) col *= 0.55;   /* contour lines */
        FragColor = finish(col * shade, n, v);
        return;
    }
    if (dot(n, v) > 0.0) n = -n;                     /* face the viewer */
    if (mode == 3 || mode == 4) {                    /* matcaps: view-space normal */
        vec3 nv = normalize(mat3(viewMatrix) * n);
        float c = (direction == 0) ? nv.x : nv.y;
        float aa = max(fwidth(c), 1e-5);
        if (mode == 3) {                             /* dark material + one bright highlight band */
            float base = 0.10 + 0.08 * (0.5 + 0.5 * nv.y) + 0.06 * pow(1.0 - nv.z, 3.0);
            float w = max(hl_width, aa);
            float d = (c - hl_pos) / w;
            float band = exp(-d * d * 4.0) + 0.18 * exp(-d * d * 0.35);
            FragColor = finish(vec3(base) + vec3(0.92, 0.94, 1.0) * band, n, v);
        } else {                                     /* grey material + thin lines */
            vec3 L = normalize(vec3(-0.35, 0.55, 0.75));
            float lam = max(dot(nv, L), 0.0);
            float g = 0.30 + 0.42 * lam + 0.10 * pow(max(nv.z, 0.0), 8.0);
            float f = c * stripes * 0.5 + 0.5;
            float dist = abs(fract(f) - 0.5) / max(fwidth(f), 1e-5);       /* pixels to the nearest line */
            float line = 1.0 - smoothstep(line_px, line_px + 1.0, dist);
            FragColor = finish(vec3(g * (1.0 - 0.75 * line)), n, v);
        }
        return;
    }
    if (mode == 5) {                                 /* environment: paint (or metal) + reflection, world-fixed */
        vec3 r = reflect(v, n);
        vec3 L = env_radiance(r, env_blur(r));
        vec3 E = textureLod(env_irr, equirect(env_frame(n)), 0.0).rgb;
        float fr = pow(1.0 - clamp(-dot(n, v), 0.0, 1.0), 5.0);
        vec3 F0 = mix(vec3(0.04), paint, metallic);  /* clear coat 4 %; a metal reflects in its own colour */
        vec3 F = F0 + (1.0 - F0) * fr;               /* Schlick */
        FragColor = vec4(tonemap(env_gain * (paint * E * (1.0 - metallic) + gloss * F * L)), opacity);
        return;
    }
    vec3 r = reflect(v, n);                          /* zebra: reflection stripes */
    vec3 rv = mat3(viewMatrix) * r;
    float c = (direction == 0) ? rv.x : rv.y;
    float f = c * stripes * 0.5;
    float px = max(fwidth(f), 1e-5);
    float tri = abs(fract(f) - 0.5) - 0.25;          /* signed distance to a stripe edge */
    float a = smoothstep(-px, px, tri);              /* crisp, antialiased edges at any zoom */
    FragColor = lin(vec4(vec3(a), opacity));
}
"""


# -- environment textures (Environment matcap) ----------------------------------

ENV_SAMPLERS = ("env_hi", "env_mid", "env_low", "env_irr")
_env_tex = {}


def _texture(img):
    """RGBA16F texture of an (h, w, 3) float image (rows bottom-up), linear
    filtering, wrapping around in u (the equirect seam)."""
    h, w = img.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = img
    tex = gpu.types.GPUTexture((w, h), format='RGBA16F', data=gpu.types.Buffer('FLOAT', w * h * 4, rgba.ravel()))
    tex.filter_mode(True)
    tex.extend_mode_x('REPEAT')
    tex.extend_mode_y('EXTEND')
    return tex


def _blank():
    hit = _env_tex.get("_blank")
    if hit is None:
        hit = _env_tex["_blank"] = _texture(np.zeros((1, 1, 3), np.float32))
    return hit


def env_textures(v):
    """(style for the shader, {sampler: texture}, gain, hi-res width) of the
    chosen environment; an HDRI that can't be read falls back to the Horizon."""
    from . import environments as env
    src = v.env_source
    if src not in ("TUNNEL", "HORIZON"):
        data = env.hdri(src)
        if data is not None:
            hit = _env_tex.get(src)
            if hit is None:
                hit = _env_tex[src] = {n: _texture(img) for n, img in zip(ENV_SAMPLERS, data["levels"] + [data["irr"]])}
            return 2, hit, data["gain"], float(data["levels"][0].shape[1])
        src = "HORIZON"
    E, gain = env.procedural(src, v.env_lights, v.env_height)
    key = (src, int(v.env_lights), round(float(v.env_height), 4))
    hit = _env_tex.get("_proc")
    if hit is None or hit[0] != key:
        hit = _env_tex["_proc"] = (key, _texture(E))
    b = _blank()
    return (0 if src == "TUNNEL" else 1), {"env_hi": b, "env_mid": b, "env_low": b, "env_irr": hit[1]}, gain, 1.0


_sag_cache = {}


def _sag(obj):
    """1.5 x the display mesh's chord tolerance in world units (the brep's box
    diagonal, as the tessellation measures it, times the object's scale)."""
    key = (obj.session_uid, obj.get(store.PROP_REV, 0))
    hit = _sag_cache.get(key)
    if hit is None:
        try:
            bb = store.get_brep(obj).bounding_box()
            diag = float(np.linalg.norm(np.array(bb[1]) - np.array(bb[0])))
        except Exception:
            diag = float(obj.dimensions.length)
        hit = 1.5 * float(store.tess_settings()[0]) * diag
        _sag_cache[key] = hit
    return hit * max(obj.matrix_world.to_scale())


def _get_shader():
    global _shader
    if _shader is not None:
        return _shader
    info = gpu.types.GPUShaderCreateInfo()
    for t, n in (('MAT4', "viewProjectionMatrix"), ('MAT4', "modelMatrix"), ('MAT4', "normalMatrix"),
                 ('MAT4', "viewMatrix"), ('VEC3', "eye"), ('VEC3', "pull"), ('FLOAT', "stripes"),
                 ('FLOAT', "opacity"), ('FLOAT', "thr"), ('FLOAT', "vmin"), ('FLOAT', "vmax"),
                 ('FLOAT', "bands"), ('FLOAT', "hl_pos"), ('FLOAT', "hl_width"), ('FLOAT', "line_px"), ('FLOAT', "bias"),
                 ('FLOAT', "env_rot"), ('FLOAT', "env_height"), ('FLOAT', "env_gain"), ('FLOAT', "env_lights"),
                 ('FLOAT', "env_texels"), ('FLOAT', "gloss"), ('FLOAT', "env_rough"), ('FLOAT', "metallic"),
                 ('VEC3', "paint"), ('INT', "direction"), ('INT', "mode"), ('INT', "style"), ('INT', "coat_on")):
        info.push_constant(t, n)
    for slot, n in enumerate(ENV_SAMPLERS):
        info.sampler(slot, 'FLOAT_2D', n)
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nor")
    info.vertex_in(2, 'FLOAT', "val")
    iface = gpu.types.GPUStageInterfaceInfo("nproto_diag_iface")
    iface.smooth('VEC3', "vP")
    iface.smooth('VEC3', "vN")
    iface.smooth('FLOAT', "vV")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "FragColor")
    info.vertex_source(_VS)
    info.fragment_source(_FS)
    _shader = gpu.shader.create_from_info(info)
    return _shader


def zebra_mesh(obj, angle_deg, chord_rel):
    """Fine tessellation of the exact B-rep (local space): points, EXACT surface
    normals at every vertex, triangle indices. Independent of the display mesh,
    whose coarse triangles would bend stripes / colours at edges."""
    pts, nrm, polys = store._tessellate(store.get_brep(obj), chord_rel, angle_deg, natural=store.is_natural(obj))
    return np.asarray(pts, np.float32), np.asarray(nrm, np.float32), _tris(polys)


def _tris(polys):
    if isinstance(polys, np.ndarray) and polys.ndim == 2 and polys.shape[1] == 3:
        return polys.astype(np.int32)
    tris = []
    for f in polys:
        f = [int(x) for x in f]
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.asarray(tris, np.int32)


def _zebra_batch(obj, shader, angle_deg, chord_rel):
    rev = obj.get(store.PROP_REV, 0)
    key = (rev, round(angle_deg, 4), round(chord_rel, 7))
    hit = _zebra_cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    try:
        P, N, T = zebra_mesh(obj, angle_deg, chord_rel)
    except Exception as exc:
        print(f"NURBS diagnostics ({obj.name}): {exc}")
        return None
    if len(T) == 0:
        return None
    batch = batch_for_shader(shader, 'TRIS', {"pos": P, "nor": N, "val": np.zeros(len(P), np.float32)}, indices=T)
    _zebra_cache[obj.session_uid] = (key, batch)
    return batch


# -- curvature ---------------------------------------------------------------

CURV_KINDS = ("GAUSS", "MEAN", "MAX", "MIN")


def curvatures_at(surf, u, v):
    """(Gaussian, mean, k1 >= k2) of a surface at (u, v), from the exact
    first / second derivatives (fundamental forms). At a singular point (a
    collapsed edge such as a sphere pole) the value is taken a tiny step inside
    the domain (the limit)."""
    from .square import _second
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    for step in (0.0, 1e-4, 1e-3, 4e-3, 1e-2):
        uu, vv = u + (uc - u) * step, v + (vc - v) * step
        su, sv, suu, suv, svv = _second(surf, uu, vv)
        E, F, G = su @ su, su @ sv, sv @ sv
        if E * G - F * F > 1e-4 * (E + G) ** 2:      # well-conditioned (not near a pole)
            break
    n = np.cross(su, sv)
    ln = np.linalg.norm(n)
    if ln < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    n /= ln
    E, F, G = su @ su, su @ sv, sv @ sv
    L, M, N = suu @ n, suv @ n, svv @ n
    den = E * G - F * F
    K = (L * N - M * M) / den
    H = (E * N - 2 * F * M + G * L) / (2 * den)
    d = max(H * H - K, 0.0) ** 0.5
    return K, H, H + d, H - d


def face_curvature_mesh(obj, angle_deg, chord_rel):
    """Per-face fine tessellation with the exact curvatures at every vertex
    (local space). Returns P, N, T, values dict {GAUSS, MEAN, MAX, MIN}."""
    from .kernel import load
    from .square import _foot
    ext = load().blender_ext
    text = obj[store.PROP_BREP]
    brep = store.get_brep(obj)
    nf = brep.face_count()
    Ps, Ns, Ts, Vs = [], [], [], {k: [] for k in CURV_KINDS}
    base = 0
    for fi in range(nf):
        if nf == 1:
            single = brep
        else:
            single = store.brep_from_text(text)
            ext.delete_faces(single, [f for i, f in enumerate(single.faces()) if i != fi])
        face = single.faces()[0]
        surf = face.surface()
        P, N, polys = store._tessellate(single, chord_rel, angle_deg)
        T = _tris(polys)
        # signs relative to the face's OUTWARD normal (convex = positive mean
        # curvature everywhere, whatever the surface's parameter direction)
        sgn = 1.0
        try:
            uv = _foot(surf, np.array(face.interior_point(), float))
            _, su, sv = surf.derivatives(*uv)
            sgn = 1.0 if float(np.cross(su, sv) @ np.array(face.outward_normal_at_uv(*uv), float)) >= 0 else -1.0
        except Exception:
            pass
        vals = []
        for uv in store.surface_feet(surf, P).tolist():
            K, H, k1, k2 = curvatures_at(surf, *uv)
            # outward normal: curvature towards the inside is positive (convex > 0)
            H, k1, k2 = -sgn * H, -sgn * k2, -sgn * k1
            vals.append((K, H, k1, k2))
        vals = np.array(vals) if vals else np.zeros((0, 4))
        Ps.append(np.asarray(P, np.float32))
        Ns.append(np.asarray(N, np.float32))
        Ts.append(T + base)
        for k, col in zip(CURV_KINDS, range(4)):
            Vs[k].append(vals[:, col] if len(vals) else np.zeros(0))
        base += len(P)
    return (np.vstack(Ps), np.vstack(Ns), np.vstack(Ts) if Ts else np.zeros((0, 3), np.int32),
            {k: np.concatenate(v) for k, v in Vs.items()})


def _curv_data(obj, angle_deg, chord_rel):
    rev = obj.get(store.PROP_REV, 0)
    key = (rev, round(angle_deg, 4), round(chord_rel, 7))
    hit = _curv_cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    try:
        data = face_curvature_mesh(obj, angle_deg, chord_rel)
    except Exception as exc:
        print(f"NURBS curvature map ({obj.name}): {exc}")
        data = None
    _curv_cache[obj.session_uid] = (key, data)
    return data


def curvature_range(objs, kind, angle_deg, chord_rel):
    """Robust (2nd..98th percentile) range over the objects; symmetric for
    Gaussian / mean curvature so zero sits in the middle of the ramp."""
    vals = []
    for o in objs:
        d = _curv_data(o, angle_deg, chord_rel)
        if d is not None and len(d[3][kind]):
            vals.append(d[3][kind])
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if kind in ("GAUSS", "MEAN"):
        m = max(abs(lo), abs(hi)) or 1e-9
        return -m, m
    if hi - lo < 1e-12:
        hi = lo + 1e-9
    return float(lo), float(hi)


_curv_batches = {}


def _curv_batch(obj, shader, kind, angle_deg, chord_rel):
    d = _curv_data(obj, angle_deg, chord_rel)
    if d is None or len(d[2]) == 0:
        return None
    key = (_curv_cache[obj.session_uid][0], kind)
    hit = _curv_batches.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    P, N, T, V = d
    batch = batch_for_shader(shader, 'TRIS', {"pos": P, "nor": N, "val": V[kind].astype(np.float32)}, indices=T)
    _curv_batches[obj.session_uid] = (key, batch)
    return batch


# -- drawing -----------------------------------------------------------------

LAST_RANGE = []          # curvature range of the last draw (shown in the panel)


def draft_angle_deg(normal, pull):
    """Draft angle (degrees) of a surface with this outward normal: + = releases
    along the pull direction, - = undercut. Same formula as the shader."""
    n = np.asarray(normal, float) / (np.linalg.norm(normal) or 1.0)
    p = np.asarray(pull, float) / (np.linalg.norm(pull) or 1.0)
    return float(np.degrees(np.arcsin(np.clip(n @ p, -1.0, 1.0))))


PULL = {"Z": (0, 0, 1), "-Z": (0, 0, -1), "X": (1, 0, 0), "-X": (-1, 0, 0), "Y": (0, 1, 0), "-Y": (0, -1, 0)}


def draw_zebra(ctx):
    """Zebra, draft angle or curvature map (whichever analysis is on)."""
    s = ctx.scene.nproto_settings
    v = ctx.window_manager.nproto_view
    if v.show_matcap:
        mode = {'HIGHLIGHT': 3, 'LINES': 4, 'ENVIRONMENT': 5}.get(v.matcap_style, 3)
    else:
        mode = 1 if v.show_draft else (2 if v.show_curvature else (0 if v.show_zebra else None))
    if mode is None:
        return
    rv3d = ctx.region_data
    if rv3d is None:
        return
    try:
        shader = _get_shader()
    except Exception as exc:
        print("NURBS diagnostics shader:", exc)
        return
    objs = [o for o in (ctx.selected_objects if v.diag_selected_only else ctx.visible_objects)
            if store.is_nurbs(o) and o.type == 'MESH']
    if not objs:
        return
    vinv = rv3d.view_matrix.inverted()
    eye = vinv.translation if rv3d.is_perspective else (vinv.translation + vinv.to_3x3() @ Vector((0, 0, 1e6)))
    if v.draft_pull == 'VIEW':
        pull = -(vinv.to_3x3() @ Vector((0, 0, -1)))        # towards the viewer
    else:
        pull = Vector(PULL[v.draft_pull])
    vmin, vmax = 0.0, 1.0
    if mode == 2:
        if v.curv_auto:
            vmin, vmax = curvature_range(objs, v.curv_kind, float(v.curv_angle), float(v.curv_chord))
            LAST_RANGE[:] = [vmin, vmax]
        else:
            vmin, vmax = float(v.curv_min), float(v.curv_max)
    coat = bool(v.env_coat) and mode in (1, 2, 3, 4)         # clear coat over draft / curvature / matcap
    style, texs, gain, texels = 0, None, 1.0, 1.0
    if mode == 5 or coat:
        try:
            style, texs, gain, texels = env_textures(v)
        except Exception as exc:
            print("NURBS environment:", exc)
            return
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("viewProjectionMatrix", rv3d.perspective_matrix)
    shader.uniform_float("viewMatrix", rv3d.view_matrix)
    shader.uniform_float("eye", eye)
    shader.uniform_float("bias", 0.003 * float(rv3d.view_distance))
    shader.uniform_float("pull", pull.normalized())
    shader.uniform_float("stripes", float(v.matcap_lines if (v.show_matcap and v.matcap_style == 'LINES')
                                          else v.zebra_stripes))
    shader.uniform_float("opacity", float(v.zebra_opacity))
    shader.uniform_float("thr", float(v.draft_angle))
    shader.uniform_float("vmin", vmin)
    shader.uniform_float("vmax", vmax)
    shader.uniform_float("bands", float(v.curv_bands))
    shader.uniform_float("hl_pos", float(v.matcap_pos))
    shader.uniform_float("hl_width", float(v.matcap_width))
    shader.uniform_float("line_px", float(v.matcap_line_px))
    b = _blank()
    for n in ENV_SAMPLERS:
        shader.uniform_sampler(n, texs[n] if texs else b)
    shader.uniform_float("env_rot", float(v.env_rotation))
    shader.uniform_float("env_height", float(v.env_height))
    shader.uniform_float("env_gain", float(gain))
    shader.uniform_float("env_lights", float(v.env_lights))
    shader.uniform_float("env_texels", texels)
    shader.uniform_float("gloss", float(v.env_gloss))
    shader.uniform_float("paint", tuple(v.env_paint))
    shader.uniform_float("env_rough", float(v.env_roughness))
    shader.uniform_float("metallic", float(v.env_metallic))
    shader.uniform_int("style", style)
    shader.uniform_int("coat_on", 1 if coat else 0)
    shader.uniform_int("direction", 0 if v.zebra_direction == 'VERTICAL' else 1)
    shader.uniform_int("mode", mode)
    for obj in objs:
        if mode == 2:
            batch = _curv_batch(obj, shader, v.curv_kind, float(v.curv_angle), float(v.curv_chord))
        else:
            batch = _zebra_batch(obj, shader, float(v.zebra_angle), float(v.zebra_chord))
        if batch is None:
            continue
        M = obj.matrix_world
        # the display mesh's chords can stand in front of the fine overlay by
        # up to its chord tolerance (hollows): pull at least that far
        shader.uniform_float("bias", max(0.003 * float(rv3d.view_distance), _sag(obj)))
        shader.uniform_float("modelMatrix", M)
        shader.uniform_float("normalMatrix", M.to_3x3().inverted_safe().transposed().to_4x4())
        batch.draw(shader)
    gpu.state.blend_set('ALPHA')
