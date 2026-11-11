"""Environments for the Environment matcap: what the surfaces reflect.

Two are drawn by the shader itself, sharp at any zoom: a Light Tunnel (long
strip lights around one horizontal axis, as in Alias) and a Horizon (bright sky
over a dark ground). The others are the HDRIs that ship with Blender
(datafiles/studiolights/world: 1K, CC0, from Poly Haven), read with
OpenImageIO, so nothing is downloaded and no image datablock is added.

Every environment comes with its irradiance (the cosine-weighted light a matte
surface receives, per normal direction) and a gain that scales the brightest
irradiance to 1, so a white matte paint never blows out and every environment
looks about as bright. An HDRI is kept at three sizes (1024, 256, 64 wide):
the shader blends toward a smaller one where the reflection sweeps over many
texels per pixel (a tight fillet), which would otherwise sparkle.

Directions map to equirect (u, v) as u = 0.5 - atan2(y, x) / 2pi,
v = 0.5 + asin(z) / pi, with v = 1 straight up (the shader uses the same)."""

import os

import numpy as np

PROCEDURAL = (("TUNNEL", "Light Tunnel", "Long strip lights around the part, sharp at any zoom (Alias style). "
                                         "Rotation turns the tunnel's axis"),
              ("HORIZON", "Horizon", "Bright sky over a dark ground with one crisp horizon line. "
                                     "Height moves the horizon"))

TUNNEL_WIDTH = 0.25        # a strip light's width, as a part of the spacing between strips
TUNNEL_WALL = 0.02         # radiance between the strips (the strips are 1)
TUNNEL_FLOOR = 0.6         # the strips below the axis are this bright


def hdri_dir():
    import bpy
    return bpy.utils.system_resource('DATAFILES', path="studiolights/world") or ""


_names = None


def hdri_names():
    """File stems of Blender's bundled world HDRIs (city, courtyard, ... studio)."""
    global _names
    if _names is None:
        d = hdri_dir()
        try:
            _names = sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith((".exr", ".hdr")))
        except OSError:
            _names = []
    return _names


def hdri_path(name):
    d = hdri_dir()
    for ext in (".exr", ".hdr"):
        p = os.path.join(d, name + ext)
        if os.path.isfile(p):
            return p
    return None


# -- equirect helpers ----------------------------------------------------------

def equirect_dirs(w, h):
    """Unit directions of the texel centres (rows bottom-up: row 0 looks down)
    and each texel's solid angle."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (0.5 - u) * 2.0 * np.pi
    lat = (v - 0.5) * np.pi
    cl = np.cos(lat)[:, None]
    D = np.stack(np.broadcast_arrays(cl * np.cos(phi)[None, :], cl * np.sin(phi)[None, :],
                                     np.sin(lat)[:, None]), axis=-1).reshape(-1, 3)
    dA = (np.cos(lat)[:, None] * np.ones(w)[None, :] * (2.0 * np.pi / w) * (np.pi / h)).reshape(-1)
    return D, dA


def equirect_uv(d):
    d = np.asarray(d, float)
    return 0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2.0 * np.pi), 0.5 + np.arcsin(np.clip(d[..., 2], -1, 1)) / np.pi


def downsample(img, f):
    """Box filter by f in both directions (sizes divisible by f)."""
    h, w, c = img.shape
    return img.reshape(h // f, f, w // f, f, c).mean(axis=(1, 3))


def irradiance(L, out_w=32, out_h=16):
    """Cosine-weighted irradiance / pi of an equirect radiance map (rows
    bottom-up), per normal direction: 1 everywhere for a uniform radiance 1."""
    h, w = L.shape[:2]
    D, dA = equirect_dirs(w, h)
    N, _ = equirect_dirs(out_w, out_h)
    cos = np.clip(N @ D.T, 0.0, None) * dA[None, :]
    return (cos @ L.reshape(-1, 3) / np.pi).reshape(out_h, out_w, 3)


# -- procedural environments (the shader draws the same, with antialiasing) ----

def tunnel(d, lights):
    """Light tunnel along the X axis: `lights` strips around it, one straight up."""
    d = np.asarray(d, float)
    s = np.hypot(d[..., 1], d[..., 2])
    phi = np.arctan2(d[..., 2], d[..., 1])
    f = (phi - np.pi / 2) / (2 * np.pi) * lights
    dist = np.abs(f - np.round(f))
    strip = (dist < TUNNEL_WIDTH / 2).astype(float)
    bright = TUNNEL_FLOOR + (1 - TUNNEL_FLOOR) * (0.5 + 0.5 * d[..., 2] / np.maximum(s, 1e-4))
    opening = np.clip((s - 0.15) / 0.2, 0.0, 1.0)            # the tunnel's open ends are dark
    L = TUNNEL_WALL + strip * bright * opening
    return np.repeat(L[..., None], 3, axis=-1)


SKY_HORIZON = np.array([1.0, 1.0, 1.0])
SKY_ZENITH = np.array([0.35, 0.45, 0.65])
GROUND_NEAR = np.array([0.03, 0.03, 0.03])
GROUND_FAR = np.array([0.01, 0.01, 0.01])


def horizon(d, height):
    d = np.asarray(d, float)
    h = d[..., 2] - height
    e = np.clip(h / max(1.0 - height, 1e-3), 0.0, 1.0)[..., None]
    g = np.clip(-h / max(1.0 + height, 1e-3), 0.0, 1.0)[..., None]
    sky = SKY_HORIZON + (SKY_ZENITH - SKY_HORIZON) * e
    ground = GROUND_NEAR + (GROUND_FAR - GROUND_NEAR) * g
    return np.where((h > 0)[..., None], sky, ground)


def _gain(E):
    lum = E @ np.array([0.2126, 0.7152, 0.0722])
    return 1.0 / max(float(lum.max()), 1e-6)


# -- the environment data the shader needs (numpy; textures are made in diagnostics) --

_data = {}


def procedural(kind, lights=8, height=0.0):
    """(irradiance 32 x 16, gain) of the Light Tunnel or the Horizon."""
    key = (kind, int(lights), round(float(height), 4))
    hit = _data.get(key)
    if hit is None:
        D, _ = equirect_dirs(256, 128)
        L = tunnel(D, int(lights)) if kind == "TUNNEL" else horizon(D, float(height))
        E = irradiance(downsample(L.reshape(128, 256, 3), 4))
        hit = _data[key] = (E.astype(np.float32), _gain(E))
    return hit


def hdri(name):
    """{"levels": [1024, 256, 64 wide radiance maps], "irr": 32 x 16, "gain"} of
    one of Blender's HDRIs (rows bottom-up), or None when it can't be read."""
    if name in _data:
        return _data[name]
    out = None
    path = hdri_path(name)
    if path:
        try:
            import OpenImageIO as oiio
            inp = oiio.ImageInput.open(path)
            if inp is None:
                raise OSError(oiio.geterror())
            try:
                spec = inp.spec()
                px = np.asarray(inp.read_image(0, 0, 0, min(3, spec.nchannels), "float"), np.float32)
            finally:
                inp.close()
            if px.ndim == 2:
                px = px[..., None]
            if px.shape[2] < 3:
                px = np.repeat(px[..., :1], 3, axis=2)
            img = np.ascontiguousarray(np.nan_to_num(np.flipud(px), nan=0.0, posinf=0.0).clip(0.0, 6e4))
            h, w = img.shape[:2]
            img = img[:h - h % 16, :w - w % 16]                   # sizes divisible by 4 x 4
            levels = [img, downsample(img, 4)]
            levels.append(downsample(levels[1], 4))
            E = irradiance(levels[2] if levels[2].shape[1] <= 128 else downsample(levels[2], 2))
            out = {"levels": levels, "irr": E.astype(np.float32), "gain": _gain(E)}
        except Exception as exc:
            print(f"NURBS environment {name}: {exc}")
    _data[name] = out
    return out
