# SatPy functions


'''


def get_l1a_satpy_scene(self) -> Scene:

    return self._generate_l1a_satpy_scene()

def get_l1b_satpy_scene(self) -> Scene:

    return self._generate_l1b_satpy_scene()

def get_l2a_satpy_scene(self) -> Scene:

    return self._generate_l2a_satpy_scene()

def get_toa_reflectance_satpy_scene(self) -> Scene:

    return self._generate_toa_reflectance_satpy_scene()

def get_chlorophyll_estimates_satpy_scene(self) -> Scene:

    return self._generate_chlorophyll_satpy_scene()

def get_products_satpy_scene(self) -> Scene:

    return self._generate_products_satpy_scene()




def _generate_satpy_scene(self) -> Scene:

    scene = Scene()

    latitudes, longitudes = self._generate_latlons()

    swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

    latitude_attrs = {
                        'file_type': None,
                        'resolution': self.resolution,
                        'standard_name': 'latitude',
                        'units': 'degrees_north',
                        'start_time': self.capture_datetime,
                        'end_time': self.capture_datetime,
                        'modifiers': (),
                        'ancillary_variables': []
                        }

    longitude_attrs = {
                        'file_type': None,
                        'resolution': self.resolution,
                        'standard_name': 'longitude',
                        'units': 'degrees_east',
                        'start_time': self.capture_datetime,
                        'end_time': self.capture_datetime,
                        'modifiers': (),
                        'ancillary_variables': []
                        }

    #scene['latitude'] = latitudes
    #scene['latitude'].attrs.update(latitude_attrs)
    #scene['latitude'].attrs['area'] = swath_def

    #scene['longitude'] = longitudes
    #scene['longitude'].attrs.update(longitude_attrs)
    #scene['longitude'].attrs['area'] = swath_def

    return scene

def _generate_latlons(self) -> tuple[xr.DataArray, xr.DataArray]:

    latitudes = xr.DataArray(self.latitudes, dims=self.dim_names_2d)
    longitudes = xr.DataArray(self.longitudes, dims=self.dim_names_2d)

    return latitudes, longitudes

def _generate_swath_definition(self) -> SwathDefinition:

    latitudes, longitudes = self._generate_latlons()
    swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

    return swath_def

def _generate_l1a_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    try:
        cube = self.l1a_cube
    except:
        return None

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            'standard_name': cube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            'units': cube.attrs['units'],
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    wavelengths = range(0,120)

    for i, wl in enumerate(wavelengths):

        data = cube[:,:,i]

        data = data.reset_coords(drop=True)
            
        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene

def _generate_l1b_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    try:
        cube = self.l1b_cube
        wavelengths = self.wavelengths
    except:
        return None

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            'standard_name': cube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            'units': cube.attrs['units'],
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for i, wl in enumerate(wavelengths):

        data = cube[:,:,i]
        
        data = data.reset_coords(drop=True)

        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene

def _generate_l2a_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    try:
        cube = self.l2a_cube
        wavelengths = self.wavelengths
    except:
        return None

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            'standard_name': cube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            'units': cube.attrs['units'],
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for i, wl in enumerate(wavelengths):

        data = cube[:,:,i]

        data = data.reset_coords(drop=True)

        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene

def _generate_toa_reflectance_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    try:
        cube = self.toa_reflectance_cube
        wavelengths = self.wavelengths
    except:
        return None

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            'standard_name': cube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            'units': cube.attrs['units'],
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for i, wl in enumerate(wavelengths):

        data = cube[:,:,i]

        data = data.reset_coords(drop=True)

        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene

def _generate_chlorophyll_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            #'standard_name': cube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            #'units': cube.attrs['units'],
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for key, chl in self.chl.items():

        name = 'chl_' + key
        scene[name] = chl
        scene[name].attrs.update(attrs)
        scene[name].attrs['standard_name'] = chl.attrs['description']
        scene[name].attrs['units'] = chl.attrs['units']
        scene[name].attrs['area'] = swath_def

    return scene

def _generate_products_satpy_scene(self) -> Scene:

    scene = self._generate_satpy_scene()
    swath_def= self._generate_swath_definition()

    attrs = {
            'file_type': None,
            'resolution': self.resolution,
            'name': None,
            'standard_name': None,
            'coordinates': ['latitude', 'longitude'],
            'units': None,
            'start_time': self.capture_datetime,
            'end_time': self.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }

    for key, product in self.products.items():

            scene[key] = product
            scene[key].attrs.update(attrs)
            scene[key].attrs['name'] = key
            scene[key].attrs['standard_name'] = key
            scene[key].attrs['area'] = swath_def

            try:
                scene[key].attrs.update(product.attrs)
            except AttributeError:
                pass


    return scene
'''