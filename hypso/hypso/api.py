"""The public API: one immutable :class:`Product` per capture and level.

Examples
--------
>>> import hypso
>>> l1d = (hypso.open("HYPSO2_HSI.20250304T103805.aeronetvenice.L1A.nc")
...            .calibrate()
...            .georeference(latitudes=lats, longitudes=lons)
...            .toa_reflectance())
>>> l1d.write()

Every processing method returns a new Product and leaves the one it was
called on unchanged. Underneath are the tree functions in
:mod:`hypso.processing` and the product tree of :mod:`hypso.io.product`
(:attr:`Product.tree`), which stay available for direct use.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.io.naming import product_name
from hypso.io.product import PRODUCT_DIMS, product_name as cube_name, read_product, write_product
from hypso.masks.flags import FLAGS_NAME, decode_flags

LEVELS = ("L1A", "L1B", "L1C", "L1D")

_GEOMETRY = ("latitude", "longitude", "sensor_zenith", "sensor_azimuth", "solar_zenith", "solar_azimuth",
             "relative_azimuth")


class Product:
    """One capture at one processing level.

    Create one with :func:`hypso.open` or from a product tree. A Product is
    immutable: processing methods return a new Product.

    Parameters
    ----------
    tree : xarray.DataTree
        A product tree as read by :func:`hypso.io.product.read_product`.

    Raises
    ------
    ValueError
        If the tree has no valid ``processing_level`` (L1A-L1D, L2A).
    """

    __slots__ = ("_tree",)

    def __init__(self, tree: xr.DataTree):
        level = tree.attrs.get("processing_level")
        if level not in LEVELS + ("L2A",):
            raise ValueError(f"not a HYPSO product tree (processing_level={level!r})")
        object.__setattr__(self, "_tree", tree)

    def __setattr__(self, name, value):
        raise AttributeError("Product is immutable; processing methods return a new Product")

    @classmethod
    def open(cls, path) -> "Product":
        """Read a product file of any level and any supported layout.

        Parameters
        ----------
        path : str or pathlib.Path
            A current-format, older grouped or ground-station L1A file.

        Returns
        -------
        Product
        """
        return cls(read_product(path))

    # -- contents -----------------------------------------------------------

    @property
    def tree(self) -> xr.DataTree:
        """The underlying product tree, as an :class:`xarray.DataTree`.

        A shallow copy: adding, replacing or removing variables and attributes
        does not change this Product, but the arrays are shared (copying the
        cube on every access would double memory), so do not modify array
        values in place - assign a new array instead.
        """
        return self._tree.copy()

    @property
    def level(self) -> str:
        """Processing level: ``"L1A"``, ``"L1B"``, ``"L1C"``, ``"L1D"`` or ``"L2A"``."""
        return self._tree.attrs["processing_level"]

    @property
    def sat_id(self) -> str:
        """Satellite, e.g. ``"HYPSO-2"``."""
        return self._tree.attrs.get("sat_id")

    @property
    def target(self) -> str:
        """Capture target name, e.g. ``"aeronetvenice"``."""
        return self._tree.attrs.get("location_description")

    @property
    def cube(self) -> xr.DataArray:
        """The product cube: ``dn``, ``Lt``, ``rhot`` or the L2A variable.

        Dimensions ``(lines, samples, bands)``, or ``(y, x, bands)`` when
        gridded, with ``wavelength``, ``radiation_wavelength`` and ``fwhm``
        band coordinates.
        """
        return self._tree.dataset[cube_name(self._tree)]

    @property
    def wavelengths(self) -> np.ndarray:
        """Band centre wavelengths in nm, as calibrated (``radiation_wavelength``)."""
        return self._tree.dataset["radiation_wavelength"].values

    @property
    def geometry(self) -> xr.Dataset:
        """Latitude, longitude and sun and sensor angles, as a Dataset.

        Empty before georeferencing (L1A, L1B). Includes the ``*_direct``
        variants when direct georeferencing was computed.
        """
        ds = self._tree.dataset
        names = [n for n in ds.data_vars
                 if n in _GEOMETRY or (n.endswith("_direct") and n[:-len("_direct")] in _GEOMETRY)]
        return ds[names]

    @property
    def flags(self) -> dict:
        """Masks decoded from ``pixel_flags``: ``{name: boolean DataArray}``.

        Empty if the product has no flags.
        """
        ds = self._tree.dataset
        return decode_flags(ds[FLAGS_NAME]) if FLAGS_NAME in ds else {}

    @property
    def variables(self) -> xr.Dataset:
        """Derived 2D products, as a Dataset.

        Those added with :meth:`with_variables` and those an atmospheric
        correction tool returns (e.g. ``chla``, ``polymer_flags``).
        """
        from hypso.io.product import _extra_variables
        ds = self._tree.dataset
        return ds[_extra_variables(ds, [cube_name(self._tree)])]

    @property
    def grid(self):
        """Name of the map grid (e.g. ``"utm33n_876m"``), or None in sensor geometry.

        See :meth:`regrid`.
        """
        return self._tree.attrs.get("grid")

    def _spatial_dims(self) -> tuple:
        from hypso.io.product import spatial_dims
        return spatial_dims(self._tree.dataset)

    @property
    def provenance(self) -> list:
        """How the product was made: one dict per processing step, oldest first.

        From ``metadata/processing``: step name, date, software version and
        git commit, input file and the step's parameters.
        """
        if "metadata" not in self._tree.children or "processing" not in self._tree["metadata"].children:
            return []
        steps = [{"step": name, **dict(node.attrs)} for name, node in self._tree["metadata/processing"].children.items()]
        return sorted(steps, key=lambda s: s.get("date_utc", ""))

    def __repr__(self):
        cube = self.cube
        return (f"<hypso.Product {self.sat_id} {self.target} {self.level}: {cube.name} "
                f"{dict(cube.sizes)}>")

    # -- pixels, spectra, geolocation --------------------------------------

    def pixel(self, latitude: float, longitude: float, direct: bool = False) -> tuple:
        """The pixel nearest to a location.

        Parameters
        ----------
        latitude, longitude : float
            Location in degrees.
        direct : bool, default False
            Use the direct (ADCS-only) georeferencing instead.

        Returns
        -------
        tuple of int
            ``(line, sample)``, or ``(y, x)`` for a gridded product.
        """
        from hypso.geometry.nearest import get_nearest_pixel
        geometry = self.geometry
        suffix = "_direct" if direct else ""
        if f"latitude{suffix}" not in geometry:
            raise ValueError(f"{self.level} product has no {'direct ' if direct else ''}geolocation")
        line, sample = get_nearest_pixel(target_latitude=latitude, target_longitude=longitude,
                                         latitudes=geometry[f"latitude{suffix}"].values,
                                         longitudes=geometry[f"longitude{suffix}"].values)
        return int(line), int(sample)

    def spectrum(self, latitude: float = None, longitude: float = None, line: int = None,
                 sample: int = None) -> xr.DataArray:
        """The cube's spectrum at one pixel.

        Parameters
        ----------
        latitude, longitude : float, optional
            Location in degrees; the nearest pixel is used.
        line, sample : int, optional
            Pixel indices, instead of a location.

        Returns
        -------
        xarray.DataArray
            Values over ``bands``, with wavelength coordinates.

        See Also
        --------
        window : Statistics of a window of pixels around a location.
        """
        if latitude is not None and longitude is not None:
            line, sample = self.pixel(latitude, longitude)
        elif line is None or sample is None:
            raise ValueError("give latitude and longitude, or line and sample")
        return self.cube.isel(dict(zip(self._spatial_dims(), (line, sample))))

    def window(self, latitude: float, longitude: float, size: int = 3, exclude_flags=None,
               cv_band_range=(405.0, 570.0)) -> xr.Dataset:
        """A window of pixels around a location and its statistics, for matchups.

        Parameters
        ----------
        latitude, longitude : float
            Location in degrees; the window is centred on the nearest pixel.
        size : int, default 3
            Window width in pixels (odd). Clipped at the image edges.
        exclude_flags : list of str, optional
            ``pixel_flags`` names (e.g. ``["land", "cloud"]``) whose pixels
            are left out.
        cv_band_range : tuple of float, default (405, 570)
            Wavelength range (nm) for ``pixel_valid`` and ``cv_median``.

        Returns
        -------
        xarray.Dataset
            Per band: ``mean``, ``median``, ``std``, ``cv`` (std / median)
            and ``valid_count``; ``pixels`` (the window) and ``excluded``.
            Attributes: centre pixel and its distance to the location,
            ``box_size``, ``pixel_valid`` and ``percent_valid`` (pixels with
            any finite band in ``cv_band_range``), ``cv_median`` and the
            window's mean angles. See :mod:`hypso.window`.
        """
        from hypso.window import window_statistics
        return window_statistics(self._tree, self.pixel(latitude, longitude), latitude, longitude, size=size,
                                 exclude_flags=exclude_flags, cv_band_range=cv_band_range)

    @property
    def resolution(self) -> float:
        """Pixel size: the larger of the mean along- and across-track ground sampling distance, in m.

        For a gridded product, the grid's pixel size in its CRS's units (m,
        or degrees for a geographic grid).
        """
        if self.grid is not None:
            return float(self._tree.attrs["grid_resolution"])
        from hypso.grid import resolution
        return resolution(self._tree)

    def swath_definition(self, direct: bool = False):
        """The product's geolocation as a :class:`pyresample.geometry.SwathDefinition`.

        Parameters
        ----------
        direct : bool, default False
            Use the direct georeferencing instead.
        """
        from hypso.grid import swath_definition
        return swath_definition(self._tree, direct=direct)

    def area_definition(self):
        """A gridded product's grid as a :class:`pyresample.geometry.AreaDefinition`.

        Raises
        ------
        ValueError
            If the product is in sensor geometry.
        """
        from hypso.grid import area_definition
        return area_definition(self._tree)

    def resample(self, target, method: str = "nearest", radius_of_influence: float = None) -> xr.Dataset:
        """Resample the product onto any target, as a bare Dataset.

        Parameters
        ----------
        target : pyresample.geometry.AreaDefinition or SwathDefinition
            A grid, or a swath such as another sensor's (for matchups).
        method : {"nearest", "bilinear", "average"}, default "nearest"
            For the cube and float variables; ``"average"`` only onto grids
            coarser than the product. Flags are combined as in
            :class:`hypso.grid.GridOperators`.
        radius_of_influence : float, optional
            Search radius in m; default 1.5 times the coarser pixel size.

        Returns
        -------
        xarray.Dataset
            The cube, derived variables, each flag as ``flag_<name>`` and
            the target's latitude and longitude.

        See Also
        --------
        regrid : The product on a map grid, as a Product.
        """
        from hypso.grid import resample_product
        return resample_product(self._tree, target, method=method, radius_of_influence=radius_of_influence)

    def regrid(self, target="utm", resolution: float = None, method: str = "nearest",
               radius_of_influence: float = None, max_gb: float = 4.0) -> "Product":
        """The product on a map grid, at the same level.

        Parameters
        ----------
        target : str, int, pyproj.CRS, AreaDefinition or Product, default "utm"
            ``"utm"`` (the WGS84 UTM zone of the scene centre), any CRS
            pyproj accepts (e.g. ``"EPSG:3035"``), a pyresample
            AreaDefinition, or a gridded Product to share its grid.
        resolution : float, optional
            Pixel size in the CRS's units. Default: this product's
            :attr:`resolution`, so nothing is oversampled. Not with an
            AreaDefinition or Product target.
        method : {"nearest", "bilinear", "average"}, default "nearest"
            ``"nearest"`` keeps one measured spectrum per pixel;
            ``"bilinear"`` blends neighbouring spectra; ``"average"`` takes
            cell means, for grids at least as coarse as the product.
        radius_of_influence : float, optional
            Search radius in m; default 1.5 times the coarser pixel size.
        max_gb : float, default 4.0
            Refuse grids whose cube would be larger.

        Returns
        -------
        Product
            Dimensions ``(y, x)`` with CF projection coordinates and a
            ``crs`` grid mapping; ``pixel_flags`` gains ``outside_swath``.

        See Also
        --------
        hypso.grid.regrid : The tree function, with the per-variable rules.
        """
        from hypso.grid import regrid
        if isinstance(target, Product):
            target = target.area_definition()
        return Product(regrid(self._tree, target, resolution=resolution, method=method,
                              radius_of_influence=radius_of_influence, max_gb=max_gb))

    def to_satpy(self, direct: bool = False):
        """The product as a Satpy Scene, one dataset per band.

        Parameters
        ----------
        direct : bool, default False
            Use the direct georeferencing for the swath.
        """
        from hypso.satpy.satpy import product_scene
        return product_scene(self, use_direct_georef=direct)

    # -- adding masks and derived products (each returns a new Product) -----

    def with_masks(self, descriptions: dict = None, **masks) -> "Product":
        """A new Product with masks added to ``pixel_flags``.

        Parameters
        ----------
        descriptions : dict, optional
            ``{name: text}`` describing masks, kept in the file.
        **masks : array_like of bool
            ``name=mask`` with the product's spatial shape, True where
            flagged. Names may contain letters, digits and ``_``.

        Returns
        -------
        Product
            Existing flags are kept; a mask with an existing name replaces it.

        Examples
        --------
        >>> flagged = l1d.with_masks(land=land, cloud=cloud)
        """
        from hypso.masks.flags import encode_flags
        merged = {name: flag.values for name, flag in self.flags.items()}
        merged.update({name: np.asarray(mask, dtype=bool) for name, mask in masks.items()})
        old_descriptions = {}
        if FLAGS_NAME in self._tree.dataset and "comment" in self._tree.dataset[FLAGS_NAME].attrs:
            for part in self._tree.dataset[FLAGS_NAME].attrs["comment"].split("; "):
                name, _, text = part.partition(": ")
                old_descriptions[name] = text
        tree = self._tree.copy()
        root = tree.to_dataset(inherit=False)
        root[FLAGS_NAME] = encode_flags(merged, {**old_descriptions, **(descriptions or {})},
                                        dims=self._spatial_dims())
        tree.dataset = root
        return Product(tree)

    def with_variables(self, **variables) -> "Product":
        """A new Product with derived 2D variables added.

        Parameters
        ----------
        **variables : array_like or xarray.DataArray
            ``name=values`` with the product's spatial shape; a DataArray's
            attributes (e.g. ``units``) are kept. Written with the product
            and read back by :func:`hypso.open`.

        Returns
        -------
        Product

        Examples
        --------
        >>> l2a.with_variables(chla=xr.DataArray(chla, attrs={"units": "mg m-3"}))
        """
        dims = self._spatial_dims()
        tree = self._tree.copy()
        root = tree.to_dataset(inherit=False)
        reserved = set(root.variables)
        for name, value in variables.items():
            if name in reserved and name not in self.variables:
                raise ValueError(f"{name!r} is already a product, geometry or flag variable")
            data = value if isinstance(value, xr.DataArray) else xr.DataArray(np.asarray(value), dims=dims)
            if data.shape != self.cube.shape[:2]:
                raise ValueError(f"{name!r} has shape {data.shape}, expected {self.cube.shape[:2]}")
            root[name] = xr.Variable(dims, data.values, dict(data.attrs))
        tree.dataset = root
        return Product(tree)

    # -- processing (each returns a new Product) ----------------------------

    def _require(self, level: str, method: str):
        if self.level != level:
            hint = f' or .to("{_NEXT[level]}")' if level in _NEXT else ""
            raise ValueError(f"{method}() needs an {level} product, this one is {self.level}{hint}")

    def calibrate(self, coeff_type: str = None, version=None, **kwargs) -> "Product":
        """Calibrate L1A to L1B: top-of-atmosphere radiance ``Lt``.

        Parameters
        ----------
        coeff_type : str, optional
            Radiometric variant: ``"abs"``, ``"moved"`` or ``"lab"``. Default:
            ``"abs"`` for HYPSO-2, ``"moved"`` for HYPSO-1.
        version : str or dict, optional
            Coefficient release, or ``{correction: release}`` to mix them,
            e.g. ``{"spectral": "V3"}``. HYPSO-2: ``"V3"`` (default), ``"V2"``,
            ``"V1"``; HYPSO-1: ``"V2"`` (default), ``"V1"``.
        **kwargs
            Passed to :func:`hypso.processing.l1a_to_l1b`: ``coeff_files``
            (explicit files), and ``radiometric``, ``smile``, ``destripe``,
            ``spectral`` (False to skip a correction).

        Returns
        -------
        Product
            L1B; the coefficients used and their checksums are recorded in
            :attr:`provenance`.
        """
        from hypso.processing import l1a_to_l1b
        self._require("L1A", "calibrate")
        return Product(l1a_to_l1b(self._tree, coeff_type=coeff_type, version=version, **kwargs))

    def georeference(self, latitudes=None, longitudes=None, direct: bool = True) -> "Product":
        """Georeference L1B to L1C: latitude, longitude and sun and sensor angles.

        Parameters
        ----------
        latitudes, longitudes : array_like or path, optional
            Indirect (GCP-corrected) pixel coordinates, as arrays or paths to
            raw float32 files. Omit both to use direct georeferencing.
        direct : bool, default True
            Also compute direct georeferencing from the ADCS telemetry
            (``*_direct`` variables).

        Returns
        -------
        Product
        """
        from hypso.processing import l1b_to_l1c
        self._require("L1B", "georeference")
        return Product(l1b_to_l1c(self._tree, latitudes=latitudes, longitudes=longitudes, direct=direct))

    def toa_reflectance(self, use_direct_georef: bool = False, ssi_source: str = "tsis",
                        use_unbinned: bool = True) -> "Product":
        """Convert L1C radiance to L1D top-of-atmosphere reflectance ``rhot``.

        Parameters
        ----------
        use_direct_georef : bool, default False
            Use the direct georeferencing's solar zenith angle.
        ssi_source : {"tsis", "thuillier"}, default "tsis"
            Solar irradiance spectrum: TSIS-1 HSRS or Thuillier (2002).
        use_unbinned : bool, default True
            Build the spectral response functions from the unbinned bands.

        Returns
        -------
        Product
            L1D, with the spectral response functions in ``metadata/srf``.
        """
        from hypso.processing import l1c_to_l1d
        self._require("L1C", "toa_reflectance")
        return Product(l1c_to_l1d(self._tree, use_direct_georef=use_direct_georef, ssi_source=ssi_source,
                                  use_unbinned=use_unbinned))

    def correct_atmosphere(self, method: str, masked: bool = False, mask_flags=None, work_dir=None,
                           config=None, **options) -> "Product":
        """Atmospheric correction to L2A.

        Parameters
        ----------
        method : {"dps", "polymer", "acolite", "ocsmart"}
            Dark pixel subtraction (in this process), or an external tool.
        masked : bool, default False
            Give the method NaN where ``pixel_flags`` are set.
        mask_flags : list of str, optional
            Only these flags, with ``masked=True``.
        work_dir : str or path, optional
            External tools: where their input and raw output go (kept);
            default a temporary directory.
        config : str, path or dict, optional
            External tools: the AC tool configuration
            (:mod:`hypso.ac.config`); default the ``HYPSO_AC_CONFIG``
            environment variable.
        **options
            Method-specific options overriding the configuration, e.g.
            ``settings={...}`` for Polymer or ACOLITE, ``output="l2r"`` for
            ACOLITE.

        Returns
        -------
        Product
            L2A: the method's reflectance on this product's bands, its 2D
            products and flags in :attr:`variables`.

        See Also
        --------
        hypso.processing.correct_atmosphere
        """
        from hypso.processing import correct_atmosphere
        return Product(correct_atmosphere(self._tree, method, masked=masked, mask_flags=mask_flags,
                                          work_dir=work_dir, config=config, **options))

    @property
    def atmospheric_correction(self):
        """Atmospheric correction method of an L2A product, e.g. ``"polymer"``; None below L2A."""
        return self._tree.attrs.get("atmospheric_correction")

    def to(self, level: str, calibrate: dict = None, georeference: dict = None,
           toa_reflectance: dict = None, correct_atmosphere: dict = None) -> "Product":
        """Run the processing steps from this level up to ``level``.

        Parameters
        ----------
        level : {"L1B", "L1C", "L1D", "L2A"}
            Target level.
        calibrate, georeference, toa_reflectance, correct_atmosphere : dict, optional
            Options for each step, passed to the method of that name. For
            L2A, ``correct_atmosphere`` must name the ``method``; there is
            no default method.

        Returns
        -------
        Product

        Examples
        --------
        >>> l1d = l1a.to("L1D", calibrate={"version": "V2"},
        ...              georeference={"latitudes": lats, "longitudes": lons})
        >>> l2a = l1d.to("L2A", correct_atmosphere={"method": "dps"})
        """
        level = level.upper()
        order = LEVELS + ("L2A",)
        if level not in order or order.index(level) < order.index(self.level):
            raise ValueError(f"cannot go from {self.level} to {level}")
        if level == "L2A" and self.level != "L2A" and not (correct_atmosphere or {}).get("method"):
            raise ValueError('to("L2A") needs correct_atmosphere={"method": ...}: one of "dps", "polymer", '
                             '"acolite", "ocsmart"')
        options = {"calibrate": calibrate or {}, "georeference": georeference or {},
                   "toa_reflectance": toa_reflectance or {}, "correct_atmosphere": correct_atmosphere or {}}
        product = self
        while product.level != level:
            step = _STEPS[product.level]
            product = getattr(product, step)(**options[step])
        return product

    # -- output -------------------------------------------------------------

    def filename(self, label: str = None, ac: str = None) -> str:
        """The product's file name (:mod:`hypso.io.naming`).

        Parameters
        ----------
        label : str, optional
            Free-text label for experiments, the last name field.
        ac : str, optional
            Override the atmospheric correction field (L2A).

        Returns
        -------
        str
            e.g. ``"HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc"``.
        """
        from hypso.processing.capture import acquisition_time
        from datetime import datetime
        time = datetime.fromisoformat(acquisition_time(self._tree.attrs))
        if ac is None and self.level == "L2A":
            ac = self.atmospheric_correction
        return product_name(self.sat_id, time, self.target, self.level, ac=ac, grid=self.grid,
                            label=label).filename()

    def write(self, path=None, out_dir=None, label: str = None, datacube: bool = False,
              significant_bits=14) -> Path:
        """Write the product in the current file format.

        Parameters
        ----------
        path : str or path, optional
            Output file. Default: :meth:`filename` in ``out_dir``.
        out_dir : str or path, optional
            Directory for the default name; the current directory if omitted.
        label : str, optional
            Label for the default name.
        datacube : bool, default False
            Write the cube as one 3D variable instead of one per band.
        significant_bits : int or None, default 14
            Keep this many significant mantissa bits of the float bands
            (L1B-L2A): about 40% smaller L1D bands at most 3.1e-5 relative
            error; None writes them lossless.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        if path is None:
            path = Path(out_dir or ".") / self.filename(label=label)
        path = Path(path)
        write_product(self._tree, path, datacube=datacube, significant_bits=significant_bits)
        return path


_STEPS = {"L1A": "calibrate", "L1B": "georeference", "L1C": "toa_reflectance", "L1D": "correct_atmosphere"}
_NEXT = {"L1A": "L1B", "L1B": "L1C", "L1C": "L1D", "L1D": "L2A"}


def open(path) -> Product:  # noqa: A001 - mirrors xarray.open_dataset / builtin-style entry point
    """Read a HYPSO product file.

    Parameters
    ----------
    path : str or pathlib.Path
        A product file of any level (L1A-L2A), in the current format, the
        older grouped format, or a ground-station L1A.

    Returns
    -------
    Product

    Examples
    --------
    >>> import hypso
    >>> l1d = hypso.open("HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc")
    """
    return Product.open(path)
