"""Exposure adjustment of RGB images: one gain for all channels, so colours
are kept (unlike a per-channel percentile stretch, which also white-balances).

Works on any RGB here: a camera image (:func:`hypso.rgb.read_camera_image`),
a camera product (:func:`hypso.rgb.camera_product`, R/G/B variables), an
:func:`hypso.rgb.HSI2RGB` rendering or any ``(..., 3)`` array.

Examples
--------
>>> brighter = adjust_exposure(camera, ev=1.5)          # +1.5 stops
>>> auto = adjust_exposure(camera)                      # 99th percentile of brightness -> 90% of full scale
>>> display = adjust_exposure(camera, gamma=2.2, normalize=True)   # 0-1 floats for plotting
"""
import numpy as np
import xarray as xr

CHANNELS = ("R", "G", "B")
#: Rec. 709 luminance weights, used for the automatic exposure.
LUMA = (0.2126, 0.7152, 0.0722)


def exposure_gain(rgb: np.ndarray, percentile: float = 99.0, target: float = 0.9, full_scale: float = 255.0) -> float:
    """The gain that maps the ``percentile`` of the image's luminance to
    ``target`` of ``full_scale`` (NaN pixels ignored)."""
    luma = np.tensordot(np.asarray(rgb, dtype=np.float64), LUMA, axes=([-1], [0]))
    level = np.nanpercentile(luma, percentile)
    if not np.isfinite(level) or level <= 0:
        raise ValueError("cannot auto-expose: the image has no positive values")
    return float(target * full_scale / level)


def black_level(rgb: np.ndarray, mode="channel", percentile: float = 1.0) -> np.ndarray:
    """Black level(s) to subtract: the ``percentile`` of each channel
    (``"channel"``: removes a colour cast such as atmospheric haze, like dark
    object subtraction) or its smallest value over the channels
    (``"common"``: keeps the colour balance)."""
    levels = np.nanpercentile(np.asarray(rgb, dtype=np.float64), percentile, axis=tuple(range(np.ndim(rgb) - 1)))
    if mode == "common":
        return np.full(3, levels.min())
    if mode == "channel":
        return levels
    raise ValueError(f"black_point must be None, 'common', 'channel' or a number, got {mode!r}")


def _adjust(rgb: np.ndarray, ev, percentile, target, full_scale, gamma, normalize, black_point, black_percentile):
    rgb = np.asarray(rgb, dtype=np.float32)
    black = np.zeros(3)
    if black_point is not None:
        black = (black_level(rgb, black_point, black_percentile) if isinstance(black_point, str)
                 else np.broadcast_to(np.asarray(black_point, dtype=np.float64), (3,)))
        rgb = np.clip(rgb - black.astype(np.float32), 0, None)
    gain = 2.0 ** ev if ev is not None else exposure_gain(rgb, percentile, target, full_scale)
    out = np.clip(rgb * np.float32(gain) / full_scale, 0.0, 1.0)
    if gamma is not None and gamma != 1.0:
        out = out ** np.float32(1.0 / gamma)
    if not normalize:
        out = out * np.float32(full_scale)
    return out, gain, black


def adjust_exposure(image, ev: float = None, percentile: float = 99.0, target: float = 0.9, gamma: float = None,
                    full_scale: float = 255.0, normalize: bool = False, black_point=None,
                    black_percentile: float = 1.0):
    """Brighten or darken an RGB image with one gain for all channels.

    Parameters
    ----------
    image : xarray.DataArray, xarray.Dataset or numpy.ndarray
        ``(y, x, channel)`` DataArray, a Dataset with ``R``, ``G``, ``B``
        variables, or an array whose last axis is R, G, B.
    ev : float, optional
        Exposure change in stops (gain 2**ev): +1 doubles, -1 halves. Default:
        automatic, so the ``percentile`` of the luminance lands at ``target``
        of ``full_scale``.
    percentile, target : float, default 99 and 0.9
        Automatic exposure only.
    gamma : float, optional
        Display gamma applied after the gain (e.g. 2.2 lifts dark water);
        None keeps the result linear.
    full_scale : float, default 255
        The value of white: 255 for the camera's 8-bit counts, 1 for 0-1
        images such as HSI2RGB output.
    normalize : bool, default False
        Return 0-1 values instead of 0-``full_scale``.
    black_point : {None, "common", "channel"} or float or 3 floats, optional
        Subtract a black level before the gain: ``"channel"`` the
        ``black_percentile`` of each channel (removes haze and the camera's
        dark offset; usually the most natural look over water), ``"common"``
        one level for all channels (keeps the colour balance), or given
        value(s) in the image's units. None (default): gain only. A channel
        that is saturated (e.g. blue in a bright summer capture) cannot be
        restored, and per-channel levels then shift the colour.
    black_percentile : float, default 1
        For ``"common"``/``"channel"``.

    Returns
    -------
    Same type as ``image``
        Float32, clipped at full scale (saturated pixels stay white). Xarray
        results carry ``exposure_gain``, ``exposure_ev``, ``exposure_gamma``
        and ``exposure_black`` attributes.
    """
    if isinstance(image, xr.Dataset):
        rgb = np.stack([image[c].values for c in CHANNELS], axis=-1)
        out, gain, black = _adjust(rgb, ev, percentile, target, full_scale, gamma, normalize, black_point,
                                  black_percentile)
        result = image.copy()
        for i, c in enumerate(CHANNELS):
            result[c] = image[c].copy(data=out[..., i])
            result[c].attrs.pop("units", None)
            result[c].attrs["long_name"] = f"RGB camera {c} channel (exposure-adjusted, for display)"
    elif isinstance(image, xr.DataArray):
        out, gain, black = _adjust(image.values, ev, percentile, target, full_scale, gamma, normalize,
                                  black_point, black_percentile)
        result = image.copy(data=out)
    else:
        result, gain, black = _adjust(image, ev, percentile, target, full_scale, gamma, normalize, black_point,
                                     black_percentile)
    if isinstance(image, (xr.Dataset, xr.DataArray)):
        result.attrs.update(exposure_gain=float(gain), exposure_ev=float(np.log2(gain)),
                            exposure_gamma=float(gamma) if gamma else 1.0,
                            exposure_black=[float(b) for b in black])
    return result
