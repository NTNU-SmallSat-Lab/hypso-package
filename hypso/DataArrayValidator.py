import numpy as np
import xarray as xr

class DataArrayValidator():

    #TODO move dataarray validators into this class

    def __init__(self, dims_shape=None, dims_names: tuple[str, str, str] =('y', 'x', 'bands'), num_dims: int=3):

        self.dims_shape = dims_shape
        self.dims_names = dims_names
        self.num_dims = num_dims

    def validate(self, data) -> xr.DataArray:

        data = self.validate_data_format(data=data)
        data = self.validate_num_dims(data=data)
        data = self.validate_shape(data=data)
        data = self.validate_dims_names(data=data)

        return data

    def validate_data_format(self, data) -> xr.DataArray:

        # Convert the data to an xarray.DataArray
        if isinstance(data, np.ndarray):
            data = self.convert_to_xarray(data)
        elif isinstance(data, xr.DataArray):
            data = data
        else:
            raise TypeError("[ERROR] Value must be a numpy ndarray or xarray DataArray.")

        return data

    def validate_num_dims(self, data: xr.DataArray):

        n = len(data.shape)

        if n != self.num_dims:
            raise ValueError(
                f"[ERROR] The data should be " + str(self.num_dims) + "-dimensional, not " + str(n) + "-dimensional."
                )

        return data


    def validate_shape(self, data: xr.DataArray) -> xr.DataArray:
            """Validate that the data matches the required dimensions and names, renaming if necessary."""
            # Check shape consistency
            if self.dims_shape:
                if data.shape[:2] != self.dims_shape:
                    raise ValueError(
                        f"[ERROR] Data shape {data.shape[:2]} does not match required dimensions {self.dims_shape}."
                    )
                
            return data

    def validate_dims_names(self, data: xr.DataArray) -> xr.DataArray:

            # Check and rename dimension names
            dims = data.dims

            # Validate the dimension names
            if len(dims) == 2:
                if dims != self.dims_names[:2]:
                    dims_names = self.dims_names[:2]
                    data = data.rename({old: new for old, new in zip(dims, dims_names)})
            elif len(dims) == 3:
                if dims != self.dims_names:
                    dims_names = self.dims_names
                    data = data.rename({old: new for old, new in zip(dims, dims_names)})
            else:
                raise ValueError("[ERROR] Data must be either 2D or 3D with proper dimension names.")

            return data

    def convert_to_xarray(self, data):
        """Convert a numpy ndarray to an xarray DataArray with specified dimensions."""
        if data.ndim == 2:
            dims = self.dims_names[:2]
            coords={self.dims_names[0]: np.arange(data.shape[0]), 
                    self.dims_names[1]: np.arange(data.shape[1]),
                    }

        elif data.ndim == 3:
            dims = self.dims_names
            coords={self.dims_names[0]: np.arange(data.shape[0]), 
                    self.dims_names[1]: np.arange(data.shape[1]),
                    self.dims_names[2]: np.arange(data.shape[2]),
                    }
        else:
            raise ValueError("[ERROR] Data must be 2D or 3D.")



        return xr.DataArray(data, coords=coords, dims=dims)
        #return xr.DataArray(data, dims=dims)