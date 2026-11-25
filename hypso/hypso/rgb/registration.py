"""Registering HYPSO-2's RGB camera image to the HSI capture (TODO 4e).

The camera image has no georeferencing. Registration is never run when an
image is read (:func:`hypso.rgb.read_camera_image` only reads and demosaics);
call :func:`register_camera` explicitly. It registers the image to the HSI
capture's own geolocation:

1. the HSI product (L1C or L1D, three bands near red, green and blue) is
   regridded onto a UTM map - a true map plane, unlike the pushbroom swath;
2. SIFT keypoints are matched between that map and the (downsampled) camera
   image (scikit-image; ratio test and cross-check);
3. RANSAC fits a homography from camera pixels to map coordinates - for a frame
   camera looking at nearly flat ground, the right model;
4. the fit is checked: enough inliers, spread over enough of the camera image
   (a fit anchored in one corner is unreliable elsewhere), and a small enough
   residual. A capture that fails is left unregistered, with the reason.

On three test captures (Ariake, Venice, Annapolis) this gave residuals of
about 190-220 m on a 150 m map, well below the HSI's 876 x 108 m pixels, in
about 40 s each (REFACTOR_PROGRESS.md ×83).

Examples
--------
>>> from hypso.rgb import find_camera_image, read_camera_image
>>> camera = read_camera_image(find_camera_image(capture_dir))
>>> reg = register_camera(camera, l1d)
>>> geo = georeference_camera(camera, reg, l1d)       # per-pixel latitude/longitude
"""
from dataclasses import dataclass, field

import numpy as np
import pyproj
import xarray as xr

# HSI bands rendered for matching (nm): close to the camera's colours.
RGB_WAVELENGTHS = (640, 550, 460)


@dataclass(frozen=True)
class CameraRegistration:
    """Result of :func:`register_camera`."""
    #: True if the fit passed the quality check.
    ok: bool
    #: Why it failed the check ("" when ok).
    reason: str
    #: 3x3 homography from full-resolution camera pixel (col, row) to map
    #: coordinates (x, y) in ``crs`` (metres); None if no fit was possible.
    homography: np.ndarray
    #: Map CRS (WKT).
    crs: str
    #: Matched keypoints and RANSAC inliers.
    matches: int
    inliers: int
    #: Residual RMS of the inliers, metres.
    rms_m: float
    #: Fraction of the camera image inside the convex hull of the inliers.
    coverage: float
    #: Settings used.
    settings: dict = field(default_factory=dict)

    def attrs(self) -> dict:
        """The registration as NetCDF attributes (provenance)."""
        return {"registration_ok": str(self.ok).lower(), "registration_reason": self.reason,
                "registration_matches": self.matches, "registration_inliers": self.inliers,
                "registration_rms_m": self.rms_m, "registration_coverage": self.coverage,
                "registration_homography": "" if self.homography is None else
                " ".join(f"{v:.10g}" for v in self.homography.ravel()),
                "registration_crs": self.crs,
                **{f"registration_{k}": v for k, v in self.settings.items()}}


def _stretch(img: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(img, [2, 98], axis=(0, 1))
    return np.nan_to_num(np.clip((img - low) / (high - low), 0, 1))


def _hsi_map(hsi, resolution: float):
    """The HSI product's three RGB bands on a UTM grid: (grey image, area)."""
    if hsi.cube.sizes["bands"] > 3:
        hsi = hsi.select_bands(wavelengths=list(RGB_WAVELENGTHS))
    grid = hsi.regrid(resolution=resolution)
    return _stretch(grid.cube.values.astype(float)).mean(axis=2), grid.area_definition()


def register_camera(camera: xr.DataArray, hsi, resolution: float = 150.0, downsample: float = 0.25,
                    min_inliers: int = 30, min_coverage: float = 0.25, max_rms_m: float = 500.0,
                    ratio: float = 0.8, ransac_threshold_px: float = 3.0, seed: int = 0) -> CameraRegistration:
    """Register a camera image to an HSI capture.

    Parameters
    ----------
    camera : xarray.DataArray
        ``(y, x, channel)`` image from :func:`hypso.rgb.read_camera_image`
        (demosaiced, rotated as the ground processing does).
    hsi : hypso.Product
        The capture's L1C or L1D product (needs geolocation).
    resolution : float, default 150
        Pixel size (m) of the UTM map the camera is matched against.
    downsample : float, default 0.25
        Scale of the camera image used for matching.
    min_inliers, min_coverage, max_rms_m
        Quality check: at least this many RANSAC inliers, covering at least
        this fraction of the camera image (convex hull), with at most this
        residual RMS (m).
    ratio : float, default 0.8
        Lowe's ratio test for descriptor matching.
    ransac_threshold_px : float, default 3
        RANSAC inlier threshold, in map pixels.
    seed : int, default 0
        RANSAC random seed (reproducible results).

    Returns
    -------
    CameraRegistration
    """
    from scipy.spatial import ConvexHull
    from skimage.feature import SIFT, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform, rescale

    settings = {"resolution_m": float(resolution), "downsample": float(downsample), "min_inliers": int(min_inliers),
                "min_coverage": float(min_coverage), "max_rms_m": float(max_rms_m), "ratio": float(ratio),
                "ransac_threshold_px": float(ransac_threshold_px)}
    map_grey, area = _hsi_map(hsi, resolution)
    crs = area.crs.to_wkt()
    image = rescale(_stretch(np.asarray(camera.values, dtype=float)), downsample, channel_axis=2, anti_aliasing=True)
    cam_grey = image.mean(axis=2)

    def failed(reason, matches=0, inliers=0, rms=float("nan"), coverage=0.0, homography=None):
        return CameraRegistration(False, reason, homography, crs, matches, inliers, rms, coverage, settings)

    sift_map, sift_cam = SIFT(), SIFT()
    try:
        sift_map.detect_and_extract(map_grey)
        sift_cam.detect_and_extract(cam_grey)
    except RuntimeError as error:                     # no keypoints at all (blank image)
        return failed(f"no keypoints: {error}")
    matches = match_descriptors(sift_cam.descriptors, sift_map.descriptors, max_ratio=ratio, cross_check=True)
    if len(matches) < 4:
        return failed(f"only {len(matches)} matches", matches=len(matches))
    src = sift_cam.keypoints[matches[:, 0]][:, ::-1]          # (col, row)
    dst = sift_map.keypoints[matches[:, 1]][:, ::-1]
    model, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=ransac_threshold_px,
                            max_trials=5000, rng=seed)
    if model is None or inliers is None or inliers.sum() < 4:
        return failed("no consistent homography", matches=len(matches))

    # full-resolution camera pixel -> downsampled pixel -> map pixel -> map coordinates
    scale = np.diag([downsample, downsample, 1.0])
    x0, y0 = area.area_extent[0], area.area_extent[3]
    to_map = np.array([[area.pixel_size_x, 0, x0 + area.pixel_size_x / 2],
                       [0, -area.pixel_size_y, y0 - area.pixel_size_y / 2], [0, 0, 1]])
    homography = to_map @ model.params @ scale
    homography /= homography[2, 2]

    n_in = int(inliers.sum())
    residual = np.sqrt(((model(src[inliers]) - dst[inliers]) ** 2).sum(axis=1)) * resolution
    rms = float(np.sqrt(np.mean(residual ** 2)))
    points = src[inliers]
    coverage = 0.0
    if n_in >= 3:
        try:
            coverage = float(ConvexHull(points).volume / (cam_grey.shape[0] * cam_grey.shape[1]))
        except Exception:                             # collinear points
            coverage = 0.0
    problems = []
    if n_in < min_inliers:
        problems.append(f"{n_in} inliers < {min_inliers}")
    if coverage < min_coverage:
        problems.append(f"inliers cover {coverage:.0%} of the image < {min_coverage:.0%}")
    if rms > max_rms_m:
        problems.append(f"residual {rms:.0f} m > {max_rms_m:.0f} m")
    return CameraRegistration(not problems, "; ".join(problems), homography, crs, len(matches), n_in, rms,
                              coverage, settings)


def _apply(homography: np.ndarray, cols, rows) -> tuple:
    h = homography
    w = h[2, 0] * cols + h[2, 1] * rows + h[2, 2]
    return (h[0, 0] * cols + h[0, 1] * rows + h[0, 2]) / w, (h[1, 0] * cols + h[1, 1] * rows + h[1, 2]) / w


def camera_lonlat(registration: CameraRegistration, shape) -> tuple:
    """Latitude and longitude (degrees, float32) of every camera pixel centre,
    ``(rows, cols)`` arrays for an image of ``shape``."""
    if registration.homography is None:
        raise ValueError(f"camera image not registered: {registration.reason}")
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    x, y = _apply(registration.homography, cols, rows)
    to_lonlat = pyproj.Transformer.from_crs(registration.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(x, y)
    return lat.astype(np.float32), lon.astype(np.float32)


def hsi_to_camera_pixels(registration: CameraRegistration, hsi, shape) -> tuple:
    """For every HSI pixel, the fractional camera ``(row, col)`` it falls on
    (via the inverse homography; exact for the model), NaN outside the
    camera image of ``shape``. Arrays of the HSI's spatial shape."""
    if registration.homography is None:
        raise ValueError(f"camera image not registered: {registration.reason}")
    lat, lon = hsi.geometry["latitude"].values, hsi.geometry["longitude"].values
    x, y = pyproj.Transformer.from_crs("EPSG:4326", registration.crs, always_xy=True).transform(lon, lat)
    cols, rows = _apply(np.linalg.inv(registration.homography), x, y)
    outside = (rows < -0.5) | (cols < -0.5) | (rows > shape[0] - 0.5) | (cols > shape[1] - 0.5)
    return np.where(outside, np.nan, rows), np.where(outside, np.nan, cols)


def camera_to_hsi_pixels(registration: CameraRegistration, hsi, shape, max_distance_m: float = None) -> tuple:
    """For every camera pixel, the nearest HSI pixel ``(line, sample)`` (-1
    where it is farther than ``max_distance_m``, default the HSI's
    resolution, i.e. outside the swath). Arrays of the camera's ``shape``."""
    from scipy.spatial import cKDTree
    lat, lon = camera_lonlat(registration, shape)
    to_map = pyproj.Transformer.from_crs("EPSG:4326", registration.crs, always_xy=True)
    hx, hy = to_map.transform(hsi.geometry["longitude"].values, hsi.geometry["latitude"].values)
    tree = cKDTree(np.column_stack([hx.ravel(), hy.ravel()]))
    cx, cy = to_map.transform(lon.astype(np.float64), lat.astype(np.float64))
    distance, index = tree.query(np.column_stack([cx.ravel(), cy.ravel()]))
    limit = hsi.resolution if max_distance_m is None else max_distance_m
    index = np.where(distance <= limit, index, -1)
    line, sample = np.divmod(index, hx.shape[1])
    line, sample = np.where(index < 0, -1, line), np.where(index < 0, -1, sample)
    return line.reshape(shape).astype(np.int32), sample.reshape(shape).astype(np.int32)


def georeference_camera(camera: xr.DataArray, registration: CameraRegistration, hsi=None) -> xr.Dataset:
    """The camera image in its own frame with per-pixel geolocation.

    Parameters
    ----------
    camera : xarray.DataArray
        ``(y, x, channel)`` image (:func:`hypso.rgb.read_camera_image`).
    registration : CameraRegistration
        From :func:`register_camera`; must be ``ok``.
    hsi : hypso.Product, optional
        The HSI product, to flag camera pixels outside its swath (their
        geolocation is extrapolated from the fit).

    Returns
    -------
    xarray.Dataset
        ``R``, ``G``, ``B`` (y, x), ``latitude``, ``longitude``, optionally
        ``in_hsi_swath``, and the registration in the attributes; writable
        with ``to_netcdf``.
    """
    if not registration.ok:
        raise ValueError(f"camera image not registered: {registration.reason}")
    shape = camera.shape[:2]
    lat, lon = camera_lonlat(registration, shape)
    data = {c: (("y", "x"), np.asarray(camera.sel(channel=c).values, dtype=np.float32),
                {"long_name": f"RGB camera {c} channel", "units": "1 (counts)"}) for c in ("R", "G", "B")}
    data["latitude"] = (("y", "x"), lat, {"standard_name": "latitude", "units": "degrees_north"})
    data["longitude"] = (("y", "x"), lon, {"standard_name": "longitude", "units": "degrees_east"})
    if hsi is not None:
        line, _ = camera_to_hsi_pixels(registration, hsi, shape)
        data["in_hsi_swath"] = (("y", "x"), (line >= 0).astype(np.uint8),
                                {"long_name": "1 where the pixel lies within the HSI swath (geolocation "
                                              "interpolated), 0 where it is extrapolated", "flag_values": [0, 1]})
    attrs = {k: v for k, v in camera.attrs.items()}
    attrs.update(registration.attrs())
    return xr.Dataset(data, attrs=attrs)
