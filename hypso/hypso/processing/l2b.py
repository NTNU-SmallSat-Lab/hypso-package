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
