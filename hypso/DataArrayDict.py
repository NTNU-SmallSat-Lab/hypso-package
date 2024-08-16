import numpy as np
import xarray as xr

class DataArrayDict(dict):
    def __init__(self, attributes=None, dims_shape=None, dim_names=('y', 'x', 'bands')):
        """
        Initialize the DataArrayDict with optional attributes and dimension shape.

        :param attributes: Dictionary of attributes to add to each xarray entry.
        :param dims_shape: Tuple specifying the shape for y and x dimensions (y_size, x_size).
        """
        self.attributes = attributes or {}
        self.dims_shape = dims_shape
        self.dim_names = dim_names
        super().__init__()

    def __setitem__(self, key, value):
        """Override the method for setting a dictionary item."""
        # Ensure key is lowercased
        key = key.lower()

        value = self.validate_data_format(data=value)
        value = self.validate_shape(data=value)
        value = self.validate_dim_names(data=value)

        value.assign_attrs(self.attributes)

        # Store the xarray.DataArray in the dictionary
        super().__setitem__(key, value)



    def validate_data_format(self, data) -> xr.DataArray:

        # Convert the data to an xarray.DataArray
        if isinstance(data, np.ndarray):
            data = self.convert_to_xarray(data)
        elif isinstance(data, xr.DataArray):
            data = data
        else:
            raise TypeError("Value must be a numpy ndarray or xarray DataArray.")

        return data

    def validate_shape(self, data: xr.DataArray) -> xr.DataArray:
            """Validate that the data matches the required dimensions and names, renaming if necessary."""
            # Check shape consistency
            if self.dims_shape:
                if data.shape[:2] != self.dims_shape:
                    raise ValueError(
                        f"Data shape {data.shape[:2]} does not match required dimensions {self.dims_shape}."
                    )
                
            return data


    def validate_dim_names(self, data: xr.DataArray) -> xr.DataArray:

            # Check and rename dimension names
            dims = data.dims

            # Validate the dimension names
            if len(dims) == 2:
                if dims != self.dim_names[:2]:
                    dim_names = self.dim_names[:2]
                    data = data.rename({old: new for old, new in zip(dims, dim_names)})
            elif len(dims) == 3:
                if dims != self.dim_names:
                    dim_names = self.dim_names
                    data = data.rename({old: new for old, new in zip(dims, dim_names)})
            else:
                raise ValueError("Data must be either 2D or 3D with proper dimension names.")

            return data

    def convert_to_xarray(self, data):
        """Convert a numpy ndarray to an xarray DataArray with specified dimensions."""
        if data.ndim == 2:
            dims = self.dim_names[:2]
        elif data.ndim == 3:
            dims = self.dim_names
        else:
            raise ValueError("Data must be 2D or 3D.")

        return xr.DataArray(data, dims=dims, attrs=self.attributes)




    def __getitem__(self, key):
        """Override to ensure lowercase key access."""
        return super().__getitem__(key.lower())

    def get(self, key, default=None):
        """Override to ensure lowercase key access."""
        return super().get(key.lower(), default)






    

    




def rename_dims_if_needed(self, xarray_data, expected_dims):
    """Rename dimensions to match expected names if they are not already correct."""
    current_dims = xarray_data.dims
    if len(current_dims) == len(expected_dims):
        rename_mapping = {dim: expected_dim for dim, expected_dim in zip(current_dims, expected_dims)}
        if set(current_dims) != set(expected_dims):
            # Rename dimensions if necessary
            xarray_data = xarray_data.rename(rename_mapping)
    return xarray_data



