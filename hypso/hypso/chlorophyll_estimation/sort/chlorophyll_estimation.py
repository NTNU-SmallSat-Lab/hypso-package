'''
# Chlorophyll estimation functions













# TODO
def load_chlorophyll_estimates(self, path: str) -> None:

    return None

def generate_chlorophyll_estimates(self, 
                                    product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                    model: Union[str, Path] = None,
                                    factor: float = 0.1
                                    ) -> None:

    self._run_chlorophyll_estimation(product_name=product_name, model=model, factor=factor)

def get_chlorophyll_estimates(self, product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                ) -> np.ndarray:

    key = product_name.lower()

    return self.chl[key]












def _run_chlorophyll_estimation(self, 
                                product_name: str, 
                                model: Union[str, Path] = None,
                                factor: float = None,
                                #overwrite: bool = False,
                                **kwargs) -> None:

    match product_name.lower():

        case "band_ratio":

            if self.VERBOSE:
                print("[INFO] Running band ratio chlorophyll estimation...")

            self.chl[product_name] = self._run_band_ratio_chlorophyll_estimation(factor=factor, **kwargs)
            
        case "6sv1_aqua":

            if self.VERBOSE:
                print("[INFO] Running 6SV1 AQUA Tuned chlorophyll estimation...")

            self.chl[product_name] = self._run_6sv1_aqua_tuned_chlorophyll_estimation(model=model, **kwargs)

        case "acolite_aqua":

            if self.VERBOSE:
                print("[INFO] Running ACOLITE AQUA Tuned chlorophyll estimation...")

            self.chl[product_name] = self._run_acolite_aqua_tuned_chlorophyll_estimation(model=model, **kwargs)

        case _:
            print("[ERROR] No such chlorophyll estimation product supported!")
            return None

    return None

# TODO: use Rrs, not L1b radiance
def _run_band_ratio_chlorophyll_estimation(self, factor: float = None) -> xr.DataArray:

    cube = self.l2a_cube.to_numpy()

    try:
        mask = self.unified_mask.to_numpy()
    except:
        mask = None

    chl = run_band_ratio_chlorophyll_estimation(cube = cube,
                                                mask = mask, 
                                                wavelengths = self.wavelengths,
                                                spatial_dimensions = self.spatial_dimensions,
                                                factor = factor
                                                )

    chl_attributes = {
                    'method': "549 nm over 663 nm band ratio",
                    'factor': factor,
                    'units': "a.u."
                    }

    chl = self._format_chl(chl)
    chl = chl.assign_attrs(chl_attributes)

    return chl

def _run_6sv1_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> xr.DataArray:

    if self.l2a_cube is None or self.l2a_cube.attrs['correction'] != '6sv1':
        self._run_atmospheric_correction(product_name='6sv1')

    model = Path(model)

    if not validate_tuned_model(model = model):
        print("[ERROR] Invalid model.")
        return None
    
    if self.spatial_dimensions is None:
        print("[ERROR] No spatial dimensions provided.")
        return None
    
    cube = self.l2a_cube.to_numpy()

    try:
        mask = self.unified_mask.to_numpy()
    except:
        mask = None

    chl = run_tuned_chlorophyll_estimation(l2a_cube = cube,
                                            model = model,
                                            mask = mask,
                                            spatial_dimensions = self.spatial_dimensions
                                            )
    
    chl_attributes = {
                    'method': "6SV1 AQUA Tuned",
                    'model': model,
                    'units': r'$mg \cdot m^{-3}$'
                    }

    chl = self._format_chl(chl)
    chl = chl.assign_attrs(chl_attributes)

    return chl

def _run_acolite_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> xr.DataArray:

    #if self.l2a_cube is None or self.l2a_cube.attrs['correction'] != 'acolite':
    #    self._run_atmospheric_correction(product_name='acolite')

    model = Path(model)

    if not validate_tuned_model(model = model):
        print("[ERROR] Invalid model.")
        return None
    
    if self.spatial_dimensions is None:
        print("[ERROR] No spatial dimensions provided.")
        return None

    cube = self.l2a_cube.to_numpy()

    try:
        mask = self.unified_mask.to_numpy()
    except:
        mask = None
    
    chl = run_tuned_chlorophyll_estimation(l2a_cube = cube,
                                            model = model,
                                            mask = mask,
                                            spatial_dimensions = self.spatial_dimensions
                                            )

    chl_attributes = {
                    'method': "ACOLITE AQUA Tuned",
                    'model': model,
                    'units': r'$mg \cdot m^{-3}$'
                    }

    chl = self._format_chl(chl)
    chl = chl.assign_attrs(chl_attributes)

    return chl

def _format_chl(self, chl: Union[np.ndarray, xr.DataArray]) -> None:

    cloud_mask_attributes = {
                            'description': "Chlorophyll estimates"
                            }
    
    v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

    data = v.validate(data=chl)
    data = data.assign_attrs(cloud_mask_attributes)

    return data
'''