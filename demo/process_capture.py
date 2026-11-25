#!/usr/bin/env python3
"""Process HYPSO captures from L1A to L2A (and L2B) with the hypso package.

The capture is a folder on the HYPSO data server (a URL such as
``http://<server>/mjosa/mjosa_2025-05-28T10-15-51Z/``) or a local capture
folder. From the server only what processing needs is downloaded: the L1A
file, the metadata and, from ``processing-temp/``, the indirect
georeferencing and the sea/land/cloud labels.

Steps: calibrate (``--release``) -> georeference (indirect when the
ground processing's GCP-corrected coordinates exist, else direct) -> TOA
reflectance -> masks (land and cloud from the ground processing's
classification, saturated pixels from the L1A counts) -> atmospheric
correction of the water pixels (``--method``) -> files. With ``--l1d-only``
it stops after writing the L1D.

Output, in ``--out/<capture>/``::

    <SAT>_HSI.<time>.<target>.L1D.nc
    <SAT>_HSI.<time>.<target>.L2A.RRS.<method>.nc       (Rrs; OC-SMART: also aod)
    <SAT>_HSI.<time>.<target>.L2B.BGC.<method>.nc       (the tool's chlorophyll-a, suspended matter)
    <SAT>_HSI.<time>.<target>.L2B.IOP.<method>.nc       (the tool's IOPs: OC-SMART's aph, adg, apg, bbp, bp)
    <SAT>_HSI.<time>.<target>.L2A.AUX.<method>.nc       (the tool's other per-band outputs:
                                                         OC-SMART's Lt, Lr, Lrc, Lwp, tg_sol, tg_sen)
    work/                                                (the AC tool's own files)

External AC tools are found through the AC tool configuration
(``--ac-config`` or the HYPSO_AC_CONFIG environment variable; see
``hypso.ac.config``). For OC-SMART use the CameronLP/OC-SMART fork's
``hypso-refactor`` branch: it reads this package's files, and its HYPSO
reader takes the land/water mask from the input's ``pixel_flags`` (water =
no flag set), so lakes are processed. Stock OC-SMART's HYPSO mask counts
only oceans as water. Example::

    ocsmart:
      dir: /path/to/OC-SMART-github
      python: /path/to/envs/ocsmart-refactor/bin/python

OC-SMART needs pre-V3 band names: use ``--release V3.1v2spec`` (default
here) or ``V1v2spec``. OC-SMART downloads its ancillary data from NASA
OceanData with the credentials in ``~/.netrc``.

Examples::

    python process_capture.py http://129.241.2.147:8009/mjosa/mjosa_2025-05-28T10-15-51Z/ \\
        --out ~/HYPSO_DATA_REFACTOR --ac-config demo/ac_tools.yaml   # a copy of demo/ac_tools.example.yaml
    python process_capture.py ~/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z --out /tmp/out
    python process_capture.py <capture> <capture> ... --out ~/HYPSO_DATA_REFACTOR
    python process_capture.py --list captures.txt --out ~/HYPSO_DATA_REFACTOR

Several captures run one after another, each into its own folder; a capture
that fails is logged and the rest still run (exit status 1 if any failed).
"""
import argparse
import faulthandler
import logging
import re
import shutil
import signal
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import hypso
from hypso.masks.sea_land_cloud import sea_land_cloud_masks

logger = logging.getLogger("process_capture")

# What processing needs from a capture folder (relative paths; {name} is the capture name).
REQUIRED = ["{name}-l1a.nc"]
OPTIONAL = ["{name}-meta.json",
            "processing-temp/latitudes_indirectgeoref.dat",
            "processing-temp/longitudes_indirectgeoref.dat",
            "processing-temp/sea-land-cloud.labels"]

#: A pixel is flagged saturated when any band reaches this fraction of full
#: scale (bin_factor x 4095 counts: the on-board binning sums detector pixels,
#: so a partly saturated bin reads below full scale).
SATURATION_FRACTION = 0.95


def capture_name(source: str) -> str:
    name = source.rstrip("/").split("/")[-1]
    if not re.match(r".+_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", name):
        raise SystemExit(f"{source!r} does not end in a capture name like mjosa_2025-05-28T10-15-51Z")
    return name


def download(url: str, path: Path) -> bool:
    """Download url to path (skipped when a file of the same size exists).
    Returns False when the server has no such file."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            size = int(response.headers.get("Content-Length") or -1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    if path.is_file() and path.stat().st_size == size:
        logger.info("have %s", path.name)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    logger.info("downloading %s (%.1f MB)", url, size / 1e6)
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if size >= 0 and partial.stat().st_size != size:
        raise IOError(f"incomplete download of {url}: {partial.stat().st_size} of {size} bytes")
    partial.rename(path)
    return True


def fetch_capture(source: str, capture_dir: Path) -> Path:
    """The capture's files in capture_dir: downloaded from a URL, or the local folder itself."""
    name = capture_name(source)
    if not re.match(r"https?://", source):
        local = Path(source).expanduser()
        if not (local / f"{name}-l1a.nc").is_file():
            raise SystemExit(f"no {name}-l1a.nc in {local}")
        return local
    base = source.rstrip("/") + "/"
    for rel in REQUIRED + OPTIONAL:
        rel = rel.format(name=name)
        found = download(base + rel, capture_dir / rel)
        if not found and rel in [r.format(name=name) for r in REQUIRED]:
            raise SystemExit(f"{base + rel} not found")
        if not found:
            logger.warning("not on the server: %s", rel)
    return capture_dir


def saturated_mask(l1a) -> np.ndarray:
    counts = l1a.cube.values
    bin_factor = int(l1a.tree["metadata/capture_config"].attrs.get("bin_factor", 1))
    return (counts >= SATURATION_FRACTION * bin_factor * 4095).any(axis=-1)


def process(name: str, capture_dir: Path, out_dir: Path, method: str, release: str, coeff_type: str,
            ac_config, l1d_only: bool = False, masked: bool = True) -> None:
    temp = capture_dir / "processing-temp"
    t0 = time.time()

    l1a = hypso.open(capture_dir / f"{name}-l1a.nc")
    logger.info("%s %s, %d x %d pixels, %d bands", l1a.sat_id, name, *l1a.cube.shape)
    saturated = saturated_mask(l1a)

    l1b = l1a.calibrate(coeff_type, version=release)
    lats, lons = temp / "latitudes_indirectgeoref.dat", temp / "longitudes_indirectgeoref.dat"
    if lats.is_file() and lons.is_file():
        l1c = l1b.georeference(latitudes=lats, longitudes=lons, direct=False)
        logger.info("indirect georeferencing (ground processing's GCP-corrected coordinates)")
    else:
        l1c = l1b.georeference()
        logger.warning("no indirect georeferencing in %s: using direct georeferencing (can be km off)", temp)
    l1d = l1c.toa_reflectance()

    masks = {"saturated": saturated}
    labels = temp / "sea-land-cloud.labels"
    if labels.is_file():
        classes = sea_land_cloud_masks(labels, l1d.shape)
        masks.update(land=classes["land"], cloud=classes["cloud"])
    else:
        logger.warning("no sea/land/cloud labels in %s: only saturated pixels are masked", temp)
    l1d = l1d.with_masks(descriptions={"saturated": f"a band at >= {SATURATION_FRACTION:.0%} of full scale"},
                         **masks)
    water = ~np.logical_or.reduce(list(masks.values()))
    logger.info("pixels: %s; left for AC: %d (%.1f%%)",
                ", ".join(f"{k} {int(v.sum())}" for k, v in masks.items()), water.sum(), 100 * water.mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("wrote %s", l1d.write(out_dir=out_dir).name)
    if l1d_only:
        logger.info("done (L1D only) in %.0f s: %s", time.time() - t0, out_dir)
        return
    if masked and not water.any():
        raise SystemExit("no unmasked pixels to correct")

    work_dir = out_dir / "work"
    if "." in str(work_dir.absolute()) and method == "ocsmart":
        raise SystemExit(f"OC-SMART cannot use a path containing '.': {work_dir.absolute()} (choose another --out)")
    if not masked:
        logger.info("unmasked: every pixel goes to %s (the masks stay in pixel_flags)", method)
    l2a = l1d.correct_atmosphere(method, masked=masked, config=ac_config, work_dir=work_dir)
    path = l2a.write(out_dir=out_dir)
    aux = l2a.intermediates()
    logger.info("wrote %s, its L2B files (%s) and %s", path.name, ", ".join(sorted(l2a.tool_products())) or "none",
                f"its L2A.AUX file ({len(aux.variables.data_vars)} variables)" if aux is not None
                else "no L2A.AUX file (no intermediates)")

    rrs = l2a.cube.values
    valid = np.isfinite(rrs).any(axis=-1)
    bands = ~np.isnan(rrs).all(axis=(0, 1))
    logger.info("%s: Rrs on %d bands for %d pixels (%.1f%% of the unmasked ones; %d of them unmasked)",
                method, int(bands.sum()), int(valid.sum()), 100 * (valid & water).sum() / max(water.sum(), 1),
                int((valid & water).sum()))
    if valid.any():
        median = np.nanmedian(rrs[valid], axis=0)
        for target in (443, 490, 560, 665, 705):
            i = int(np.nanargmin(np.where(bands, np.abs(l2a.wavelengths - target), np.inf)))
            logger.info("  median Rrs %5.1f nm: %.5f sr-1", l2a.wavelengths[i], median[i])
    logger.info("done in %.0f s: %s", time.time() - t0, out_dir)


def read_capture_list(path: str) -> list:
    """Captures from a text file: one URL or folder per line; blank lines and # comments are skipped."""
    lines = Path(path).expanduser().read_text().splitlines()
    return [line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("capture", nargs="*",
                        help="capture URL(s) on the data server, or local capture folder(s)")
    parser.add_argument("--list", metavar="FILE", default=None,
                        help="text file of captures, one URL or folder per line (# comments allowed); "
                             "added after any given on the command line")
    parser.add_argument("--out", default=".", help="output root; files go to <out>/<capture>/ (default: .)")
    parser.add_argument("--method", default="ocsmart", help="AC method (default ocsmart; also machi, dps, "
                                                             "polymer, acolite)")
    parser.add_argument("--release", default="V3.1v2spec",
                        help="calibration release (default V3.1v2spec, the V2 band names OC-SMART needs)")
    parser.add_argument("--coeff-type", default="abs", help="radiometric variant (default abs)")
    parser.add_argument("--ac-config", default=None,
                        help="AC tool configuration YAML (default: $HYPSO_AC_CONFIG)")
    parser.add_argument("--l1d-only", action="store_true",
                        help="stop after the L1D (TOA reflectance with masks); no atmospheric correction")
    parser.add_argument("--unmasked", action="store_true",
                        help="give the AC tool every pixel, not only the unmasked ones (the masks are still "
                             "written as pixel_flags in the outputs); OC-SMART's input then has no "
                             "pixel_flags, so it uses its own HYPSO water mask (global_land_mask's is_ocean: "
                             "every lake is land, so lake-only scenes such as Erie get nothing)")
    args = parser.parse_args(argv)
    # `kill -USR1 <pid>` prints every thread's Python stack to stderr (for diagnosing a hang).
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    captures = list(args.capture) + (read_capture_list(args.list) if args.list else [])
    if not captures:
        parser.error("give at least one capture, or --list FILE")
    for source in captures:          # check every name before processing anything
        capture_name(source)

    # One at a time; a failed capture is logged and the rest still run.
    failed = []
    for i, source in enumerate(captures, 1):
        name = capture_name(source)
        if len(captures) > 1:
            logger.info("=== capture %d of %d: %s", i, len(captures), name)
        try:
            out_dir = Path(args.out).expanduser() / name
            capture_dir = fetch_capture(source, out_dir / "input")
            process(name, capture_dir, out_dir, args.method, args.release, args.coeff_type, args.ac_config,
                    l1d_only=args.l1d_only, masked=not args.unmasked)
        except KeyboardInterrupt:
            raise
        except BaseException as error:   # includes SystemExit from a capture with nothing to correct
            if len(captures) == 1:
                raise
            logger.exception("%s failed: %s", name, error)
            failed.append((name, error))
    if len(captures) > 1:
        logger.info("=== %d of %d captures done%s", len(captures) - len(failed), len(captures),
                    "".join(f"\n  FAILED {name}: {error}" for name, error in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
