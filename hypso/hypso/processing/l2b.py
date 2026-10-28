"""L2A to L2B: geophysical products in their own files.

An L2B product holds products derived from one L2A product, one group per
file (:data:`hypso.io.naming.L2B_GROUPS`: ``CHL``, ``IOP``, ``TSM``, ``KD``):
2D variables only, no band cube; per-wavelength quantities are separate
variables (``aph_443``, ``bbp_443``), as in NASA's OC_IOP files. It keeps
the L2A's geometry, ``pixel_flags``, grid and metadata, so it can be
regridded, windowed and read by Satpy like any other level. File names:
``...L2B.<GROUP>.<source>.nc``, where the source is the algorithm (e.g.
``oc4``) or the AC tool whose own product it is (e.g. ``polymer``).
"""
import re

import numpy as np
import xarray as xr

from hypso.io.naming import L2B_GROUPS, _check_token
from hypso.io.product import GRID_DIMS, GRID_MAPPING, _GEOMETRY_ATTRS, spatial_dims
from hypso.io.provenance import add_step
from hypso.masks.flags import FLAGS_NAME


def make_l2b(tree: xr.DataTree, group: str, source: str, variables: dict, parameters: dict = None) -> xr.DataTree:
    """An L2B tree from an L2A tree and the derived variables.

    Parameters
    ----------
    tree : xarray.DataTree
        The L2A product the variables were derived from.
    group : str
        Product group, a key of :data:`hypso.io.naming.L2B_GROUPS`.
    source : str
        Algorithm or tool, e.g. ``"oc4"`` or ``"polymer"`` (a file-name token).
    variables : dict
        ``{name: xarray.DataArray or (array, attrs)}``, each of the product's
        spatial shape; attributes should include ``units`` and ``long_name``.
    parameters : dict, optional
        Recorded in the ``l2b_<group>_<source>`` provenance step.

    Returns
    -------
    xarray.DataTree
    """
    if tree.attrs.get("processing_level") != "L2A":
        raise ValueError(f"L2B products are derived from L2A, got {tree.attrs.get('processing_level')}")
    if group not in L2B_GROUPS:
        raise ValueError(f"unknown L2B group {group!r}; known: {sorted(L2B_GROUPS)}")
    _check_token("source", source)
    if not variables:
        raise ValueError("no variables")
    root = tree.to_dataset(inherit=False)
    dims = spatial_dims(root)
    keep = [n for n in root.data_vars if n in _GEOMETRY_ATTRS or n in ("crs_wgs84", GRID_MAPPING, FLAGS_NAME)]
    out = root[keep]
    out = out.drop_vars([c for c in out.coords if c not in GRID_DIMS])
    shape = tuple(root.sizes[d] for d in dims)
    for name, value in variables.items():
        data, attrs = (value.values, dict(value.attrs)) if isinstance(value, xr.DataArray) else value
        data = np.asarray(data)
        if data.shape != shape:
            raise ValueError(f"{name!r} has shape {data.shape}, expected {shape}")
        if name in out:
            raise ValueError(f"{name!r} is a geometry or flag variable")
        out[name] = xr.Variable(dims, data, attrs)
    out.attrs = dict(root.attrs)
    out.attrs.update(processing_level="L2B", l2b_group=group, l2b_source=source)

    result = tree.copy()
    result.dataset = out
    input_source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, f"l2b_{group.lower()}_{source}", input_file=input_source or "L2A (in memory)",
             parameters={"group": group, "source": source, "variables": " ".join(variables), **(parameters or {})})
    return result


# The AC tools' own geophysical products, by variable name, and their L2B group.
# Anything else an AC tool returns (e.g. Polymer's logfb, Rgli, Rnir) describes
# the correction and stays in the L2A. The tools' flag variables (`*_flags`)
# stay in the L2A and are also copied into each L2B, for quality screening.
_TOOL_PRODUCT_GROUPS = (
    ("CHL", re.compile(r"^(chla|chl_.*|chlor_a.*|ndci)$", re.IGNORECASE)),
    ("TSM", re.compile(r"^(spm.*|tsm.*|tur_.*|t_dogliotti.*|t_nechad.*)$", re.IGNORECASE)),
    ("KD", re.compile(r"^kd.*$", re.IGNORECASE)),
)


def tool_product_group(name: str):
    """The L2B group of an AC tool's variable (e.g. ``"chla"`` -> ``"CHL"``),
    or None if it stays in the L2A."""
    for group, pattern in _TOOL_PRODUCT_GROUPS:
        if pattern.match(name):
            return group
    return None


def split_tool_products(tree: xr.DataTree) -> tuple:
    """Separate an L2A's AC-tool products from its reflectance.

    Parameters
    ----------
    tree : xarray.DataTree
        An L2A tree, e.g. from :func:`hypso.processing.correct_atmosphere` or
        an L2A file written before the products moved to L2B.

    Returns
    -------
    tuple
        ``(l2a_tree, {group: l2b_tree})``: the L2A without the products, and
        one L2B tree per group (:data:`hypso.io.naming.L2B_GROUPS`), named after
        the AC tool (``L2B.CHL.polymer``). The tool's flag variables are kept
        in the L2A and copied into each L2B.
    """
    from hypso.io.naming import name_method
    from hypso.io.product import _extra_variables, has_cube, product_name
    if tree.attrs.get("processing_level") != "L2A":
        raise ValueError(f"tool products are split from an L2A, got {tree.attrs.get('processing_level')}")
    root = tree.to_dataset(inherit=False)
    names = _extra_variables(root, [product_name(tree)] if has_cube(tree) else [])
    groups = {}
    for name in names:
        group = tool_product_group(name)
        if group:
            groups.setdefault(group, []).append(name)
    if not groups:
        return tree, {}
    flags = [n for n in names if n.endswith("_flags")]
    source = name_method(tree.attrs["atmospheric_correction"])
    l2b = {group: make_l2b(tree, group, source,
                           {n: root[n] for n in members + flags},
                           parameters={"from": "AC tool output", "tool_flags": " ".join(flags)})
           for group, members in groups.items()}
    moved = [n for members in groups.values() for n in members]
    rest = tree.copy()
    rest.dataset = root.drop_vars(moved)
    return rest, l2b


def derive(tree: xr.DataTree, group: str, algorithm: str = None) -> xr.DataTree:
    """An L2B tree from one of the package's algorithms on an L2A.

    Parameters
    ----------
    tree : xarray.DataTree
        An L2A tree; most algorithms need ``Rrs`` (``L2A.RRS.*``).
    group : str
        L2B group, e.g. ``"CHL"`` or ``"TSM"``.
    algorithm : str, optional
        See :data:`hypso.processing.l2b_algorithms.ALGORITHMS`; default the
        group's default (CHL: ``"oci"``, TSM: ``"nechad2010"``).

    Returns
    -------
    xarray.DataTree
        ``...L2B.<group>.<algorithm>``.
    """
    from hypso.io.product import product_name
    from hypso.processing.l2b_algorithms import ALGORITHMS, default_algorithm
    if tree.attrs.get("processing_level") != "L2A":
        raise ValueError(f"L2B products are derived from L2A, got {tree.attrs.get('processing_level')}")
    algorithm = algorithm or default_algorithm(group)
    if (group, algorithm) not in ALGORITHMS:
        raise ValueError(f"no algorithm {algorithm!r} for {group}; available: "
                         f"{sorted(a for g, a in ALGORITHMS if g == group)}")
    function, inputs, reference, _ = ALGORITHMS[(group, algorithm)]
    name = product_name(tree)
    if name not in inputs:
        raise ValueError(f"{group} {algorithm} needs an L2A with {' or '.join(inputs)}, this one has {name} "
                         f"(e.g. derive from an L2A.RRS product: Polymer, OC-SMART, ACOLITE L2W)")
    cube = tree.dataset[name]
    outputs = function(cube.values, cube["radiation_wavelength"].values)
    variables = {var: (values.astype(np.float32), {**attrs, "algorithm": algorithm, "reference": reference})
                 for var, (values, attrs) in outputs.items()}
    return make_l2b(tree, group, algorithm, variables,
                    parameters={"algorithm": algorithm, "reference": reference, "input_variable": name})
