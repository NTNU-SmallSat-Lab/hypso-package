#!/usr/bin/env python3
"""Footprint shapefiles, RGB images and maps for processed HYPSO captures.

Takes capture folders written by ``process_capture.py`` (an L1D and any
L2A/L2B files) and writes, with :mod:`hypso.maps`::

    <capture>/footprint/<capture>_footprint.shp (.shx .dbf .prj .cpg)
                                                   the swath outline, one feature
    <capture>/rgb/rgb_L1D.png, rgb_L1D_bin3.png    RGB in sensor geometry (server orientation;
                                                   bin3: samples averaged by 3, like the servers')
    <capture>/rgb/rgb_L2A.<Q>.<method>.png         RGB of each L2A (Rrs or rhos)
    <capture>/maps/map_rgb_L1D.png                 RGB on a UTM grid with the footprint,
                                                   coastlines, graticule, scale bar
    <capture>/maps/map_rgb_L2A.<Q>.<method>.png
    <capture>/maps/map_<Q>_<nm>_L2A.<Q>.<method>.png   one band of each L2A (--wavelength, default 560)
    <capture>/maps/map_<variable>_L2B.BGC.<method>.png each concentration in each L2B.BGC

and with ``--overview DIR``, per target and for all captures together::

    DIR/<target>_footprints.shp, DIR/<target>_overview.png
    DIR/all_footprints.shp,      DIR/all_overview.png
    DIR/<name>_footprints.shp,   DIR/<name>_overview.png   for each --group NAME=PREFIX
                                                         (targets starting with PREFIX)

Example::

    ./make_maps.py ~/HYPSO_DATA_REFACTOR/koreanCalVal* --overview ~/HYPSO_DATA_REFACTOR/overview \
        --group korea=koreanCalVal

Needs shapely, pyshp, matplotlib and (for coastlines) cartopy with its
Natural Earth data.
"""
import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import hypso  # noqa: E402
from hypso import maps  # noqa: E402

logger = logging.getLogger("make_maps")
PRODUCT = re.compile(r"\.(L1D|L2A\.(RRS|RHOS|RHOT)\.[^.]+|L2B\.BGC\.[^.]+)\.nc$")


def capture_files(folder: Path) -> dict:
    """{product label: path} for the L1D, L2A cube files and L2B.BGC files in folder."""
    files = {}
    for path in sorted(folder.glob("*.nc")):
        match = PRODUCT.search(path.name)
        if match:
            files[match.group(1)] = path
    return files


def save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", path.name)


def run_capture(folder: Path, wavelength: float, dpi: int, skip_existing: bool):
    files = capture_files(folder)
    if "L1D" not in files:
        logger.warning("%s: no L1D, skipped", folder.name)
        return None
    capture = folder.name
    t0 = time.time()
    logger.info("== %s: %s", capture, ", ".join(files))

    def todo(path):
        return not (skip_existing and path.is_file())

    l1d_rgb = hypso.open(files["L1D"], wavelengths=list(maps.RGB_WAVELENGTHS))
    shp = folder / "footprint" / f"{capture}_footprint.shp"
    if todo(shp):
        maps.write_footprints(l1d_rgb, shp)
        logger.info("  wrote %s", shp.name)
    for name, kwargs in (("rgb_L1D.png", {}), ("rgb_L1D_bin3.png", {"bin_samples": 3})):
        if todo(folder / "rgb" / name):
            maps.save_rgb(l1d_rgb, folder / "rgb" / name, **kwargs)
            logger.info("  wrote %s", name)
    path = folder / "maps" / "map_rgb_L1D.png"
    if todo(path):
        fig, ax = plt.subplots(figsize=(7, 8))
        maps.plot_map(l1d_rgb, ax=ax)
        save(fig, path, dpi)

    for label, file in files.items():
        if label.startswith("L2A"):
            quantity = label.split(".")[1]
            name = {"RRS": "Rrs", "RHOS": "rhos", "RHOT": "rhot"}[quantity]
            rgb = hypso.open(file, wavelengths=list(maps.RGB_WAVELENGTHS))
            if todo(folder / "rgb" / f"rgb_{label}.png"):
                maps.save_rgb(rgb, folder / "rgb" / f"rgb_{label}.png")
                logger.info("  wrote rgb_%s.png", label)
            path = folder / "maps" / f"map_rgb_{label}.png"
            if todo(path):
                fig, ax = plt.subplots(figsize=(7, 8))
                maps.plot_map(rgb, ax=ax)
                save(fig, path, dpi)
            path = folder / "maps" / f"map_{name}_{wavelength:g}_{label}.png"
            if todo(path):
                band = hypso.open(file, wavelengths=[wavelength])
                fig, ax = plt.subplots(figsize=(7, 8))
                maps.plot_map(band, name, wavelength, ax=ax)
                save(fig, path, dpi)
        elif label.startswith("L2B"):
            bgc = hypso.open(file)
            for variable in bgc.variables.data_vars:
                if variable.endswith("flags") or bgc.variables[variable].dtype.kind != "f":
                    continue
                path = folder / "maps" / f"map_{variable}_{label}.png"
                if todo(path):
                    fig, ax = plt.subplots(figsize=(7, 8))
                    maps.plot_map(bgc, variable, ax=ax)
                    save(fig, path, dpi)
    logger.info("   %s done in %.0f s", capture, time.time() - t0)
    return l1d_rgb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("captures", nargs="+", help="capture folders (from process_capture.py)")
    parser.add_argument("--overview", metavar="DIR", default=None,
                        help="also write footprint shapefiles and overview maps per target and for all")
    parser.add_argument("--group", action="append", default=[], metavar="NAME=PREFIX",
                        help="with --overview: also one overview of all targets starting with PREFIX (repeatable)")
    parser.add_argument("--wavelength", type=float, default=560.0, help="band of the L2A maps (nm, default 560)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--skip-existing", action="store_true", help="keep outputs that already exist")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    by_target, failed = defaultdict(list), []
    for folder in map(Path, args.captures):
        try:
            product = run_capture(folder.expanduser(), args.wavelength, args.dpi, args.skip_existing)
        except Exception:
            logger.exception("%s failed", folder.name)
            failed.append(folder.name)
            continue
        if product is not None:
            by_target[product.target].append(product)

    if args.overview and by_target:
        out = Path(args.overview).expanduser()
        groups = dict(by_target)
        if len(groups) > 1:
            groups["all"] = [p for products in by_target.values() for p in products]
        for spec in args.group:
            name, _, prefix = spec.partition("=")
            members = [p for target, products in by_target.items() if target.startswith(prefix or name)
                       for p in products]
            if members:
                groups[name] = members
            else:
                logger.warning("--group %s: no captures with a target starting with %r", name, prefix or name)
        for name, products in groups.items():
            maps.write_footprints(products, out / f"{name}_footprints.shp")
            fig, ax = plt.subplots(figsize=(8, 8))
            maps.plot_overview(products, ax=ax, title=f"{name}: {len(products)} HYPSO captures")
            save(fig, out / f"{name}_overview.png", args.dpi)
    if failed:
        logger.error("failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
