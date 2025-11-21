import numpy as np
import xarray as xr

from .DataArrayValidator import DataArrayValidator

class DataArrayDict(dict, DataArrayValidator):
    def __init__(self, 
                 attributes: dict = None, 
                 dim_shape: tuple[int, int, int] = None, 
                 dim_names: tuple[str, str, str] = ('y', 'x', 'bands'), 
                 num_dims: int = 2, 
                 key_attribute: str = None):
        """
        Initialize the DataArrayDict with optional attributes and dimension shape.

        :param attributes: Dictionary of attributes to add to each xarray entry.
        :param dims_shape: Tuple specifying the shape for y, x, and band dimensions (y_size, x_size, bands_size).
        :param dim_names: Tuple specifying the names for y, x, and band dimensions (y_name, x_name, bands_name).
        :param num_dims: Integer specifying number of dimensions (2 or 3).
        :param key_attribute: Optional string specifying the attribute to store the key as in each xarray entry. 
        """

        self.key_attribute = key_attribute

        self.attributes = attributes or {}
        self.dim_shape = dim_shape
        self.dim_names = dim_names
        self.num_dims = num_dims

        super().__init__()

    def __setitem__(self, key, value):
        """Override the method for setting a dictionary item."""
        
        # Ensure key is lowercased
        key = key.lower()

        try:
            v = DataArrayValidator(dims_shape=self.dim_shape, dim_names=self.dim_names, num_dims=self.num_dims)
            value = v.validate(data=value)
            value = value.assign_attrs(self.attributes)

            # If the 'key_attribute' is set then assign the DataArrayDict key to the specified atttribute for the xarray entry
            if self.key_attribute is not None:
                value.attrs[self.key_attribute] = key

        except Exception as ex:
            print(ex)


        

        # Store the xarray.DataArray in the dictionary
        super().__setitem__(key, value)


    def __getitem__(self, key):
        """Override to ensure lowercase key access."""
        return super().__getitem__(key.lower())

    def get(self, key, default=None):
        """Override to ensure lowercase key access."""
        return super().get(key.lower(), default)







    








