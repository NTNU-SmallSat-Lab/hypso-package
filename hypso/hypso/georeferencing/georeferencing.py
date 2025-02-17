
from pyproj import Transformer
import rasterio
import csv
import skimage
import numpy as np
import os 

# TODO check if current image_mode and destination image mode are the same 
# TODO add indication in file name in case of cube origin mode

class GCPList(list):

    SUPPORTED_IMAGE_MODES = ['bin3', 'scale3', 'standard']
    SUPPORTED_ORIGIN_MODES = ['qgis', 'cube']

    def __init__(self, filename, crs='epsg:4326', image_mode=None, origin_mode=None, cube_height=None, cube_width=None):

        super().__init__()

        # Split the path from the filename
        path, file = os.path.split(filename)

        # Convert to string if they aren't already
        path = str(path)
        file = str(file)

        # Drop the file extension
        base, extension = file.rsplit('.', 1)

        # Find image mode string and remove it from basename
        image_mode_indices = []
        for im in self.SUPPORTED_IMAGE_MODES:
            image_mode_indices.append(base.find('-' + im))

        if max(image_mode_indices) < 1:
            basename = base
        else:
            basename = base[:max(image_mode_indices)]

        # Set filename info
        self.filename = str(filename)
        self.path = str(path)
        self.extension = str(extension)
        self.basename = str(basename)

        # Set CRS
        self.crs = crs

        # Set image mode
        self.image_mode = self._check_image_mode(image_mode)

        # Set origin mode:
        self.origin_mode = self._check_origin_mode(origin_mode)

        # Set height:
        self.cube_height = cube_height

        # Set width:
        self.cube_width = cube_width


        self._load_gcps()

    
    def _detect_image_mode(self):

        detected_image_mode = 'bin3'

        for image_mode in self.SUPPORTED_IMAGE_MODES:
            if '-' + image_mode in str(self.filename):
                detected_image_mode = image_mode
            
        if detected_image_mode:    
            print('No image mode provided. Detected image mode: ' + detected_image_mode)
        else:
            print('No image mode provided. Assuming image mode is: ' + detected_image_mode)

        return detected_image_mode

    def _check_image_mode(self, image_mode):

        if not image_mode:
            image_mode = self._detect_image_mode()

        if image_mode not in self.SUPPORTED_IMAGE_MODES:
            print('Invalid image mode ' + image_mode + ' provided. Defaulting to \'standard\' image mode.')
            image_mode = 'standard'

        return image_mode


    def _check_origin_mode(self, origin_mode):

        if not origin_mode:
            origin_mode = None
            return origin_mode

        if origin_mode not in self.SUPPORTED_ORIGIN_MODES:
            print('Invalid origin mode ' + origin_mode + ' provided.')
            origin_mode = None

        return origin_mode


    def _load_gcps(self):

        header, fieldnames, unproc_gcps = PointsCSV(self.filename).read_points_csv()

        for unproc_gcp in unproc_gcps:

            gcp = GCP(**unproc_gcp, crs=self.crs)

            self.append(gcp)


    def save(self, filename=None):

        if filename is None:
            print('Writing .points file to: ' + self.filename)
            pcsv = PointsCSV(self.filename)
        else:
            pcsv = PointsCSV(filename)

        pcsv.write_points_csv(gcps=self)



    def _update_filename(self):

        match self.image_mode:

            case 'standard':
                self.filename = str(self.path) + '/' + str(self.basename) + '.' + str(self.extension)

            case 'bin3':
                self.filename = str(self.path) + '/' + str(self.basename) + '-bin3.' + str(self.extension)

            case 'scale3':
                self.filename = str(self.path) + '/' + str(self.basename) + '-scale3.' + str(self.extension)

            case _:
                print('Invalid image_mode')


    def convert_crs(self, dst_crs=None):

        if dst_crs is not None:
        
            for gcp in self:

                gcp.convert_gcp_crs(dst_crs)

    
    def change_image_mode(self, dst_image_mode=None):

        match self.image_mode:

            case 'standard':

                match dst_image_mode:

                    case 'standard':
                        self._update_filename()

                    case 'bin3':
                        self._standard_to_bin3_image_mode()
                        self._update_filename()

                    case 'scale3':
                        self._standard_to_scale3_image_mode()
                        self._update_filename()

                    case _:
                        print('Invalid dst_image_mode')

            case 'bin3':
                
                match dst_image_mode:

                    case 'standard':
                        self._bin3_to_standard_image_mode()
                        self._update_filename()

                    case 'bin3':
                        self._update_filename()

                    case 'scale3':
                        self._bin3_to_scale3_image_mode()
                        self._update_filename()

                    case _:
                        print('Invalid dst_image_mode')

            case 'scale3':

                match dst_image_mode:

                    case 'standard':
                        self._scale3_to_standard_image_mode()
                        self._update_filename()

                    case 'bin3':
                        self._scale3_to_bin3_image_mode()
                        self._update_filename()

                    case 'scale3':
                        self._update_filename()

                    case _:
                        print('Invalid dst_image_mode')

            case _:
                print('Invalid image_mode')

    # standard image mode conversion functons

    def _standard_to_bin3_image_mode(self):

        for idx, gcp in enumerate(self):

            # Apply binning
            gcp['sourceY'] = gcp['sourceY'] / 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'bin3'

    def _standard_to_scale3_image_mode(self):

        for idx, gcp in enumerate(self):

            # Apply scaling
            gcp['sourceX'] = gcp['sourceX'] * 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'scale3'

    # bin3 image mode conversion functions

    def _bin3_to_standard_image_mode(self):
                
        for idx, gcp in enumerate(self):

            # Apply scaling
            gcp['sourceY'] = gcp['sourceY'] * 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'standard'

    def _bin3_to_scale3_image_mode(self):
        
        for idx, gcp in enumerate(self):

            # Apply scaling
            gcp['sourceX'] = gcp['sourceX'] * 3
            gcp['sourceY'] = gcp['sourceY'] * 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'scale3'

    # scale3 image mode conversion functions

    def _scale3_to_standard_image_mode(self):
        
        for idx, gcp in enumerate(self):

            # Apply scaling
            gcp['sourceX'] = gcp['sourceX'] / 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'standard'

    def _scale3_to_bin3_image_mode(self):
        
        for idx, gcp in enumerate(self):

            # Apply scaling
            gcp['sourceX'] = gcp['sourceX'] / 3
            gcp['sourceY'] = gcp['sourceY'] / 3

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.image_mode = 'bin3'


    def change_origin_mode(self, dst_origin_mode=None):


        if self.cube_height is None or self.cube_width is None:

            print('No available cube height or width information. Unable to change origin mode.')

            return

        match self.origin_mode:

            case 'qgis':

                # convert to cube origin mode
                self._qgis_to_cube_origin_mode()

            case 'cube':

                # convert to qgis origin mode
                self._cube_to_qgis_origin_mode()

            case _:

                print('No origin mode set. Please first provide an origin mode before running this function.')



    def _qgis_to_cube_origin_mode(self):

        image_mode_height = self._get_image_mode_height()

        for idx, gcp in enumerate(self):

            # Switch to top left origin
            gcp['sourceY'] = gcp['sourceY'] + image_mode_height

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.origin_mode = 'cube'


    def _cube_to_qgis_origin_mode(self):

        image_mode_height = self._get_image_mode_height()

        for idx, gcp in enumerate(self):

            # Switch to top left origin
            gcp['sourceY'] = gcp['sourceY'] - image_mode_height

            # Update GCP
            self[idx] = GCP(**gcp, crs=gcp.crs)

        self.origin_mode = 'cube'

    def _get_image_mode_height(self):

        match self.image_mode:

            case 'standard':
                return self.cube_height

            case 'bin3':
                return self.cube_height / 3

            case 'scale3':
                return self.cube_height

            case _:
                print('Invalid image_mode')
                return self.cube_height

    def _get_image_mode_width(self):
        match self.image_mode:

            case 'standard':
                return self.cube_width

            case 'bin3':
                return self.cube_width

            case 'scale3':
                return self.cube_width * 3

            case _:
                print('Invalid image_mode')
                return self.cube_width

class GCP(dict):

    def __init__(self, mapX, mapY, sourceX, sourceY, enable=1, dX=0, dY=0, residual=0, crs='epsg:4326'):

        # Initialize dict
        super().__init__(mapX=mapX,
                         mapY=mapY,
                         sourceX=sourceX,
                         sourceY=sourceY,
                         enable=enable,
                         dX=dX,
                         dY=dY,
                         residual=residual)

        self.crs=crs

        # Add rasterio GCP
        self.gcp = rasterio.control.GroundControlPoint(row=self['sourceX'],
                                                        col=self['sourceY'], 
                                                        x=self['mapX'], 
                                                        y=self['mapY'])


    def convert_gcp_crs(self, dst_crs):

        src_crs = self.crs

        if src_crs.lower() != dst_crs.lower():

            # Initialize transformer for CRS conversion
            transformer = Transformer.from_crs(src_crs, dst_crs)

            # mapX is lon
            # mapY is lat

            lon = self['mapX']
            lat = self['mapY']

            lat, lon = transformer.transform(lon, lat)

            self['mapX'] = lon
            self['mapY'] = lat

            # Update rasterio GCP
            self.gcp = rasterio.control.GroundControlPoint(row=self['sourceX'],
                                                            col=self['sourceY'], 
                                                            x=self['mapX'], 
                                                            y=self['mapY'])

            self.crs = dst_crs


class PointsCSV():

    def __init__(self, filename):

        self.filename = str(filename)
        
        self.default_header = '#CRS: GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
        self.default_fieldnames = 'mapX,mapY,sourceX,sourceY,enable,dX,dY,residual'
        self.default_fieldnames_list = ['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual']



    def write_points_csv(self, gcps=[]):
        
        with open(str(self.filename), 'w') as csv_file:

            csv_file.write(self.default_header)
            csv_file.write('\n')

            # Open CSV file for writing
            writer = csv.DictWriter(csv_file, 
                                    fieldnames=self.default_fieldnames_list)
            
            writer.writeheader()

            if len(gcps) > 0:

                for gcp in gcps:
                    writer.writerow(gcp)


            # Close file
            csv_file.close()

    def read_points_csv(self):

        # Unprocessed GCPs list
        unproc_gcps = []

        with open(str(self.filename), 'r') as csv_file:

            header = csv_file.readline().rstrip()
            fieldnames = csv_file.readline().rstrip()

            # Open CSV file for reading
            reader = csv.DictReader(csv_file, 
                                    fieldnames=self.default_fieldnames_list)

            # Iterate through rows
            for line in reader:

                # Convert to int/floats
                for key, value in line.items():
                    try:
                        line[key] = int(value)
                    except ValueError:
                        try:
                            line[key] = float(value)
                        except ValueError:
                            pass

                # Add line to unprocessed GCPs list
                unproc_gcps.append(line)

            # Close file
            csv_file.close()

        return header, fieldnames, unproc_gcps 





class Georeferencer(GCPList):

    def __init__(self, filename, cube_height, cube_width, crs='epsg:4326', image_mode=None, origin_mode='qgis'):
        
        super().__init__(filename, crs=crs, image_mode=image_mode, origin_mode=origin_mode, cube_height=cube_height, cube_width=cube_width)

        self.cube_height = cube_height
        self.cube_width = cube_width

        self.img_coords = None
        self.geo_coords = None

        self.latitudes = None
        self.longitudes = None

        # Estimate polynomial transform
        self._estimate_polynomial_transform()

        # Generate latitude and longitude arrays
        self._generate_polynomial_lat_lon_arrays()


    def _estimate_polynomial_transform(self):

        # https://scikit-image.org/docs/stable/api/skimage.transform.html#polynomialtransform

        self.img_coords = np.zeros((len(self),2))
        self.geo_coords = np.zeros((len(self),2))

        # Load image coords and geospatial coords from GCPs.
        for i,gcp in enumerate(self):

            self.img_coords[i,0] = gcp['sourceX'] + 0.5
            self.img_coords[i,1] = -gcp['sourceY']*3 - 0.5

            self.geo_coords[i,0] = gcp['mapX']
            self.geo_coords[i,1] = gcp['mapY']
        
        # Estimate transform
        self.transform = skimage.transform.estimate_transform('polynomial', self.img_coords, self.geo_coords, 2)

        # Get coefficients from transform
        self.lat_coefficients = self.transform.params[0]
        self.lon_coefficients = self.transform.params[1]




    def _generate_polynomial_lat_lon_arrays(self):

            # Create empty arrays to write lat and lon data
            self.latitudes = np.empty((self.cube_height, self.cube_width))
            self.longitudes = np.empty((self.cube_height, self.cube_width))

            # Generate X and Y coordinates
            x_coords, y_coords = np.meshgrid(np.arange(self.cube_height), np.arange(self.cube_width), indexing='ij')

            # Combine the X and Y coordinates into a list of (x, y) tuples
            image_coordinates = list(zip(x_coords.ravel(), y_coords.ravel()))

            # Transform X and Y coordnates to geospatial coordinates
            geo_coordinates = self.transform(image_coordinates)

            # Copy transformed lat and lon coords into lat and lon arrays
            for idx,coord in enumerate(image_coordinates):
                self.longitudes[coord] = geo_coordinates[idx,0]
                self.latitudes[coord] = geo_coordinates[idx,1]












