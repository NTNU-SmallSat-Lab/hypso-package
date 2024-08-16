import numpy as np
import xarray as xr

class DataArrayDict(dict):
    def __init__(self, attributes=None, dims_shape=None):
        """
        Initialize the DataArrayDict with optional attributes and dimension shape.

        :param attributes: Dictionary of attributes to add to each xarray entry.
        :param dims_shape: Tuple specifying the shape for y and x dimensions (y_size, x_size).
        """
        self.attributes = attributes or {}
        self.dims_shape = dims_shape
        super().__init__()

    def __setitem__(self, key, value):
        """Override the method for setting a dictionary item."""
        # Ensure key is lowercased
        key = key.lower()

        # Convert the value to an xarray.DataArray
        if isinstance(value, np.ndarray):
            value = self.convert_to_xarray(value)
        elif isinstance(value, xr.DataArray):
            self.validate_dims(value)
            value = value.assign_attrs(self.attributes)
        else:
            raise TypeError("Value must be a numpy ndarray or xarray DataArray.")

        # Store the xarray.DataArray in the dictionary
        super().__setitem__(key, value)

    def convert_to_xarray(self, data):
        """Convert a numpy ndarray to an xarray DataArray with specified dimensions."""
        if data.ndim == 2:
            dims = ('y', 'x')
        elif data.ndim == 3:
            dims = ('y', 'x', 'bands')
        else:
            raise ValueError("Data must be 2D or 3D.")

        self.validate_dims(data)

        return xr.DataArray(data, dims=dims, attrs=self.attributes)

    def validate_dims(self, data):
        """Validate that the data matches the required dimensions."""
        if self.dims_shape:
            if data.shape[:2] != self.dims_shape:
                raise ValueError(
                    f"Data shape {data.shape[:2]} does not match required dimensions {self.dims_shape}."
                )

    def __getitem__(self, key):
        """Override to ensure lowercase key access."""
        return super().__getitem__(key.lower())

    def get(self, key, default=None):
        """Override to ensure lowercase key access."""
        return super().get(key.lower(), default)


