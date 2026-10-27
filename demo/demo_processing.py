# %% [markdown]
# # HYPSO processing demo
#
# One capture from L1A to L1D, atmospheric correction, masks and derived
# products, and map grids, with the `hypso.Product` API. Cells are
# separated by `# %%`, so this file runs as a script or cell by cell in VS
# Code / Jupyter; `demo_processing.ipynb` is generated from it
# (`python make_notebook.py`).
#
# Set the paths below to a capture of your own. The reference capture used in
# the package tests is a HYPSO-2 wide capture over Venice (2025-03-04).
# The figures need matplotlib (`pip install matplotlib`), which the package
# itself does not depend on.

# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hypso

# Calibration and regridding log what they do (coefficients used, grid size).
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

CAPTURE_DIR = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z")
L1A_FILE = CAPTURE_DIR / "aeronetvenice_2025-03-04T10-38-05Z-l1a.nc"
# Indirect (GCP-corrected) pixel coordinates from the ground processing.
LATITUDES = CAPTURE_DIR / "processing-temp" / "latitudes_indirectgeoref.dat"
LONGITUDES = CAPTURE_DIR / "processing-temp" / "longitudes_indirectgeoref.dat"
OUT_DIR = Path("demo_output")
OUT_DIR.mkdir(exist_ok=True)

SITE = (45.3139, 12.5083)   # AAOT, the AERONET-OC platform off Venice

# %% [markdown]
# ## L1A to L1D
#
# Each step returns a new, immutable `Product`; the one it was called on is
# unchanged. `hypso.open` reads any level and any supported file layout.

# %%
l1a = hypso.open(L1A_FILE)
print(l1a)

l1b = l1a.calibrate()                                   # radiometric "abs", V3 coefficients (defaults)
l1c = l1b.georeference(latitudes=LATITUDES, longitudes=LONGITUDES)
l1d = l1c.toa_reflectance()
print(l1d)

# The same in one call, running only the missing steps:
# l1d = l1a.to("L1D", georeference={"latitudes": LATITUDES, "longitudes": LONGITUDES})

# Other calibration choices (docs/calibration.rst). V3 uses V2 spectral
# coefficients by default, for OC-SMART; to use V3 spectral too:
# l1b = l1a.calibrate(version={"spectral": "V3"})
# The pre-V3 "moved" set:
# l1b = l1a.calibrate("moved", version="V2")

# %% [markdown]
# ## Writing, file names and provenance
#
# Files are named `{SAT}_HSI.{time}.{target}.{LEVEL}[.{ac}][.{grid}][.{label}].nc`.
# How a product was made is recorded in the file (`metadata/processing`).

# %%
path = l1d.write(out_dir=OUT_DIR)
print(path.name)
print(l1d.write(out_dir=OUT_DIR, label="test").name)    # experiments: an optional label

for step in l1d.provenance:
    print(step["step"], step.get("software_version"), step.get("coefficient_release", ""))

reread = hypso.open(path)
# bands are stored with 14 significant bits by default (write(significant_bits=None): lossless)
assert np.allclose(reread.cube.values, l1d.cube.values, rtol=5e-5)

# %% [markdown]
# ## Contents: cube, wavelengths, geometry, spectra

# %%
print("cube", dict(l1d.cube.sizes), l1d.cube.dtype)
print("wavelengths", l1d.wavelengths[:3], "...", l1d.wavelengths[-1])
print("geometry", list(l1d.geometry.data_vars))
print(f"resolution {l1d.resolution:.0f} m")

line, sample = l1d.pixel(*SITE)
spectrum = l1d.spectrum(*SITE)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
band = int(np.argmin(np.abs(l1d.wavelengths - 550)))
axs[0].imshow(l1d.cube.values[:, :, band], cmap="gray")
axs[0].plot(sample, line, "r+", ms=12, mew=2)
axs[0].set_title(f"L1D rhot, {l1d.wavelengths[band]:.0f} nm")
axs[1].imshow(l1d.geometry["solar_zenith"].values)
axs[1].set_title("solar_zenith")
axs[2].plot(spectrum["wavelength"], spectrum.values)
axs[2].set(title=f"rhot at AAOT (pixel {line}, {sample})", xlabel="wavelength (nm)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Matchup windows
#
# `window()` is `spectrum()` for matchups: the size x size pixels around the
# site's nearest pixel and their per-band statistics (mean, median, std, CV =
# std / median), as hypso-processing-pipeline computes them. Attributes hold
# the centre pixel and its distance to the site, the valid pixel count,
# cv_median over 405-570 nm and the window's mean angles.

# %%
w3 = l1d.window(*SITE, size=3)
w5 = l1d.window(*SITE, size=5)
print({k: w5.attrs[k] for k in ("box_size", "center_distance_m", "pixel_valid", "cv_median", "mean_solar_zenith")})

# Leave flagged pixels out: l1d.window(*SITE, size=5, exclude_flags=["land", "cloud"])

fig, axs = plt.subplots(1, 2, figsize=(13, 4))
zoom = 12
red, green, blue = (int(np.argmin(np.abs(l1d.wavelengths - w))) for w in (640, 550, 460))
crop = l1d.cube.values[line - zoom:line + zoom + 1, sample - zoom:sample + zoom + 1][:, :, [red, green, blue]]
low, high = np.nanpercentile(crop, [2, 98], axis=(0, 1))
crop = np.clip((crop - low) / (high - low), 0, 1)
axs[0].imshow(crop, extent=[sample - zoom - 0.5, sample + zoom + 0.5, line + zoom + 0.5, line - zoom - 0.5],
              aspect="auto")
for w, colour in ((w3, "yellow"), (w5, "red")):
    half = int(w.attrs["box_size"].split("x")[0]) / 2
    axs[0].add_patch(plt.Rectangle((sample - half, line - half), 2 * half, 2 * half, fill=False, ec=colour,
                                   lw=1.5, label=w.attrs["box_size"]))
axs[0].plot(sample, line, "w+", ms=10)
axs[0].legend()
axs[0].set(title="windows around AAOT (pixels are ~876 x 108 m)", xlabel="sample", ylabel="line")
for w, colour in ((w3, "orange"), (w5, "red")):
    axs[1].plot(w["radiation_wavelength"], w["mean"], color=colour, label=f"{w.attrs['box_size']} mean")
    axs[1].fill_between(w["radiation_wavelength"], w["mean"] - w["std"], w["mean"] + w["std"], color=colour,
                        alpha=0.2)
axs[1].plot(spectrum["wavelength"], spectrum.values, "k:", label="centre pixel")
axs[1].set(title="rhot, window mean +/- std", xlabel="wavelength (nm)")
axs[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RGB
#
# `HSI2RGB` renders the cube through the CIE colour matching functions.

# %%
from hypso.rgb import HSI2RGB

rgb = HSI2RGB(l1d.wavelengths, l1d.cube.values)
plt.figure(figsize=(6, 4))
plt.imshow(rgb, aspect="auto")
plt.title("L1D, HSI2RGB")
plt.show()

# %% [markdown]
# ## RGB camera
#
# HYPSO-2's RGB camera image (`<capture id>-rgb/*.bmp`, ~30 m pixels) has no
# georeferencing. `register_camera` matches it to the HSI image (SIFT +
# RANSAC homography); it is an explicit step, never run on reading. A
# registered camera product has latitude/longitude for every camera pixel,
# can be regridded onto a map and is written as
# `HYPSO2_RGB.<time>.<target>.L1A.nc` (Satpy reader `hypso_rgb_l1a`). Not
# every capture has a camera image (this cell skips if there is none).

# %%
from hypso.rgb import (camera_product, find_camera_image, plot_rgb_map, read_camera_image, regrid_camera,
                       register_camera, write_camera)

image_file = find_camera_image(CAPTURE_DIR)
if image_file is None:
    print("no RGB camera image in", CAPTURE_DIR)
else:
    camera = read_camera_image(image_file)
    reg = register_camera(camera, l1d)
    print("registered:", reg.ok, reg.reason, f"{reg.inliers} inliers, RMS {reg.rms_m:.0f} m")
    if reg.ok:
        camera_l1a = camera_product(camera, reg, l1d)
        print(write_camera(camera_l1a, out_dir=OUT_DIR).name)
        plot_rgb_map(regrid_camera(camera_l1a), graticule=True, scale_bar=True, north_arrow=True,
                     coastlines=True)
        plt.show()

# %% [markdown]
# ## Masks and derived products
#
# Products are written unmasked. Masks go into the CF flag variable
# `pixel_flags` (ESA SNAP shows them as masks); derived 2D products are
# written alongside the cube.

# %%
water = l1d.cube.values[:, :, int(np.argmin(np.abs(l1d.wavelengths - 800)))] < 0.05
masked = l1d.with_masks(land=~water, descriptions={"land": "rhot(800 nm) >= 0.05"})
ndvi_bands = [int(np.argmin(np.abs(l1d.wavelengths - w))) for w in (670, 800)]
red, nir = (l1d.cube.values[:, :, i] for i in ndvi_bands)
with_ndvi = masked.with_variables(ndvi=((nir - red) / (nir + red)).astype(np.float32))
print(list(with_ndvi.flags), list(with_ndvi.variables.data_vars))

# %% [markdown]
# ## Atmospheric correction
#
# MACHI, the default, runs in this process: one scattering and one
# transmission spectrum for the whole image, fitted so the corrected spectra
# are smooth; it gives surface reflectance `rhos` (not water-leaving Rrs).
# `dps` (dark pixel subtraction) also runs in this process. Polymer, ACOLITE
# and OC-SMART run in their own installations, configured in a YAML file (see
# docs/atmospheric_correction.rst); `masked=True` gives them NaN where the
# flags are set.

# %%
machi = with_ndvi.correct_atmosphere()                  # MACHI
print(machi, machi.filename())
model = machi.tree["metadata/machi"].dataset            # the fitted spectra
l2a = with_ndvi.correct_atmosphere("dps", masked=True, mask_flags=["land"])
print(l2a, l2a.filename())

# l2a = l1d.correct_atmosphere("polymer", config="ac_tools.yaml")
# l2a = l1c.correct_atmosphere("acolite", config="ac_tools.yaml", output="l2w")
# l2a = l1d.correct_atmosphere("ocsmart", config="ac_tools.yaml")
# l2a.variables    # the tool's 2D products and flags, e.g. chla, polymer_flags

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(model["wavelength"], model["T"], label="T (transmission)")
axs[0].plot(model["wavelength"], model["A"], label="A (scattering)")
axs[0].set(title="MACHI fitted spectra", xlabel="wavelength (nm)")
axs[0].legend()
axs[1].plot(machi.wavelengths, machi.spectrum(*SITE).values / np.pi, label="MACHI rhos / pi")
axs[1].plot(l2a.wavelengths, l2a.spectrum(*SITE).values, label="DPS Rrs")
axs[1].set(title="at AAOT", xlabel="wavelength (nm)", ylabel="sr-1")
axs[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Band subsets
#
# From L1B on, `select_bands` keeps some bands (by range, wavelengths or
# index); `hypso.open(path, wavelength_range=...)` reads only those from a
# file. Each band keeps its index in the full band set, so later steps give
# the same values as on all bands.

# %%
visible = l1d.select_bands(wavelength_range=(450, 700))
print(dict(visible.cube.sizes), visible.cube["band_index"].values[[0, -1]])
three = hypso.open(path, wavelengths=[640, 550, 460])
print(three.wavelengths)

# %% [markdown]
# ## Map grids
#
# `regrid()` puts a product (L1C, L1D or L2A) on a map grid as a product of
# the same level: by default the capture's UTM zone at its own resolution,
# nearest neighbour, so every pixel keeps one measured spectrum. Grid pixels
# the capture does not cover are flagged `outside_swath`.

# %%
grid = l1d.regrid()
print(grid, grid.grid)
print(grid.write(out_dir=OUT_DIR).name)

fine = l1d.regrid(resolution=300)                       # finer UTM grid
coarse = l1d.regrid(resolution=3000, method="average")  # 3 km cell means
geographic = l1d.regrid("EPSG:4326", resolution=0.01)

def quick_rgb(product, wavelengths=(640, 550, 460)):
    """Three bands, each stretched between its 2nd and 98th percentile
    (NaN, i.e. outside the capture, shown white)."""
    bands = [int(np.argmin(np.abs(product.wavelengths - w))) for w in wavelengths]
    image = product.cube.values[:, :, bands].astype(float)
    low, high = np.nanpercentile(image, [2, 98], axis=(0, 1))
    image = np.clip((image - low) / (high - low), 0, 1)
    return np.where(np.isnan(image), 1.0, image)


def extent_km(product):
    x, y = product.area_definition().get_proj_vectors()
    return [x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3]


fig, axs = plt.subplots(1, 3, figsize=(15, 6))
for ax, product, title in ((axs[0], fine, "nearest, 300 m"), (axs[1], coarse, "average, 3 km")):
    ax.imshow(quick_rgb(product), extent=extent_km(product))
    ax.set(title=f"{product.grid} ({title})", xlabel="easting (km)", ylabel="northing (km)")
count = coarse.variables["source_pixel_count"].values.astype(float)
count[count == 0] = np.nan
plt.colorbar(axs[2].imshow(count, extent=extent_km(coarse)), ax=axs[2], shrink=0.7)
axs[2].set(title="swath pixels per 3 km cell", xlabel="easting (km)")
plt.tight_layout()
plt.show()

# Another capture onto the same grid, for pixel-by-pixel comparison:
# other_on_same_grid = other_l1d.regrid(grid)

# %% [markdown]
# ## Resampling onto anything else
#
# `resample()` returns a bare xarray Dataset on any pyresample target,
# including another sensor's swath (e.g. for matchups).

# %%
from pyresample.geometry import AreaDefinition

area = AreaDefinition("aaot", "around AAOT", "aaot", "EPSG:4326", 80, 60,
                      (SITE[1] - 0.2, SITE[0] - 0.15, SITE[1] + 0.2, SITE[0] + 0.15))
around_site = l1d.resample(area)
print(around_site)

# %% [markdown]
# ## Satpy
#
# `to_satpy()` gives a Satpy Scene; installed, the package also registers
# Satpy readers for the product files (`hypso_l1a` ... `hypso_l1d`).

# %%
scene = l1d.to_satpy()
print(len(scene.keys()), "bands")
