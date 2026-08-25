"""Dataset-backed keyed cube/mask containers.

DatasetDict supersedes DataArrayDict (hypso/DataArrayDict.py) for
HypsoBase._l2a_cubes and HypsoBase._custom_masks. DataArrayDict had three
defects this class exists to fix:

1. Validation failed silently - its __setitem__ caught every exception,
   *printed* it, and stored the unvalidated value anyway. Here validation
   raises, so a shape/dimension mistake fails at the assignment that caused
   it, not later in whatever consumed the bad entry.
2. It subclassed dict directly, so update()/setdefault()/|= bypassed the
   __setitem__ override (no validation), and only __getitem__/get lowercased
   keys ('Rrs' in d was False while d['Rrs'] worked). This class implements
   collections.abc.MutableMapping, whose derived methods all funnel through
   the four core methods, so every path validates and every path lowercases.
3. It multiply-inherited DataArrayValidator but never used the inheritance
   (it instantiated a fresh validator inside __setitem__).

Backing the entries with a real xarray.Dataset (instead of a plain dict of
DataArrays) is what makes this the standard/generalizable choice: xarray
enforces dimension consistency across entries natively, per-entry attrs are
ordinary DataArray attrs, and the whole collection serializes with
.dataset.to_netcdf() / converts with .dataset for any cross-entry xarray
operation - none of which the hand-rolled dict could do.

DataArrayDict itself is NOT deleted: HypsoBase._products still uses it, and
the products/_products surface is intentionally-untouched forward-looking
infrastructure (user instruction, see REFACTOR_PROGRESS.md). Migrating it here
can happen whenever that surface is next worked on.
"""
from collections.abc import MutableMapping

import numpy as np
import xarray as xr


class DatasetDict(MutableMapping):
    """A dict-style view over an xarray.Dataset holding same-shaped DataArrays.

    Mapping semantics match the DataArrayDict it supersedes: keys are
    lowercased on every access, numpy arrays are converted to DataArrays with
    the configured dimension names, `attributes` are stamped onto every entry,
    and `key_attribute` (if set) stores each entry's key in that attr (e.g.
    l2a cubes carry attrs['correction'] = "polymer").

    :param attributes: attrs stamped onto every entry at assignment.
    :param dim_shape: if set, entries' leading two dimensions must have this
        shape (a capture's spatial_dimensions); a mismatch raises ValueError.
    :param dim_names: dimension names entries are normalized to.
    :param num_dims: required dimensionality of every entry (2 or 3).
    :param key_attribute: optional attr name to store each entry's key under.
    """

    def __init__(self,
                 attributes: dict = None,
                 dim_shape: tuple = None,
                 dim_names: tuple = ('y', 'x', 'bands'),
                 num_dims: int = 2,
                 key_attribute: str = None):
        self.attributes = dict(attributes or {})
        self.dim_shape = tuple(dim_shape) if dim_shape is not None else None
        self.dim_names = tuple(dim_names)
        self.num_dims = num_dims
        self.key_attribute = key_attribute
        self._ds = xr.Dataset()

    # --- validation/conversion (raises on failure, never stores bad data) ---

    def _as_dataarray(self, value) -> xr.DataArray:
        if isinstance(value, np.ndarray):
            dims = self.dim_names[:value.ndim]
            if value.ndim not in (2, 3):
                raise ValueError(f"Data must be 2D or 3D, not {value.ndim}D.")
            value = xr.DataArray(
                value,
                dims=dims,
                coords={d: np.arange(n) for d, n in zip(dims, value.shape)},
            )
        elif not isinstance(value, xr.DataArray):
            raise TypeError(
                f"Value must be a numpy ndarray or xarray DataArray, "
                f"not {type(value).__name__}."
            )

        if len(value.dims) != self.num_dims:
            raise ValueError(
                f"Data must be {self.num_dims}-dimensional, "
                f"not {len(value.dims)}-dimensional."
            )

        expected = self.dim_names[:len(value.dims)]
        if tuple(value.dims) != expected:
            value = value.rename(dict(zip(value.dims, expected)))

        if self.dim_shape is not None and value.shape[:2] != self.dim_shape:
            raise ValueError(
                f"Data shape {value.shape[:2]} does not match required "
                f"spatial dimensions {self.dim_shape}."
            )

        # Guard against xarray's assignment alignment: Dataset.__setitem__
        # REINDEXES an incoming array whose indexed dims disagree with the
        # Dataset's existing sizes - silently truncating/NaN-padding the data
        # rather than erroring (verified against xarray in this environment).
        # An explicit check turns that silent data mangling into the loud
        # failure this container exists to provide.
        for dim, size in self._ds.sizes.items():
            if dim in value.sizes and value.sizes[dim] != size:
                raise ValueError(
                    f"Dimension {dim!r} has size {value.sizes[dim]}, but this "
                    f"container's existing entries have {dim}={size} - all "
                    f"entries must share the same dimensions."
                )

        return value

    # --- MutableMapping core (everything else derives from these) ---

    def __setitem__(self, key, value):
        key = key.lower()
        value = self._as_dataarray(value)
        value = value.assign_attrs(self.attributes)
        if self.key_attribute is not None:
            value.attrs[self.key_attribute] = key
        self._ds[key] = value

    def __getitem__(self, key) -> xr.DataArray:
        # The returned DataArray wraps the Dataset's own Variable, so mutating
        # its .attrs (e.g. cube.attrs['l2_variable_name'] = ...) persists -
        # same contract callers relied on with the plain-dict predecessor.
        return self._ds[key.lower()]

    def __delitem__(self, key):
        del self._ds[key.lower()]

    def __iter__(self):
        return iter(self._ds.data_vars)

    def __len__(self):
        return len(self._ds.data_vars)

    def __repr__(self):
        return f"{type(self).__name__}({list(self._ds.data_vars)})"

    # --- extras ---

    @property
    def dataset(self) -> xr.Dataset:
        """The backing xarray.Dataset - for serialization
        (d.dataset.to_netcdf(...)) or cross-entry xarray operations."""
        return self._ds

    def copy(self) -> "DatasetDict":
        """Container-level copy: the new DatasetDict has an independent entry
        registry and independent per-entry attrs, while the underlying data
        arrays are shared (never mutated in place anywhere in this package -
        same aliasing rationale as HypsoBase._spawn_next_level)."""
        new = type(self)(attributes=self.attributes,
                         dim_shape=self.dim_shape,
                         dim_names=self.dim_names,
                         num_dims=self.num_dims,
                         key_attribute=self.key_attribute)
        new._ds = self._ds.copy(deep=False)
        return new
