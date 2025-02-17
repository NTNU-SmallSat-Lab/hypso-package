# todo: Implement logging
# import logging
__version__ = "1.5.15"


# try:
#     from osgeo import gdal
#
#     # just imports DWWaterDetect if gdal is present
#
# except BaseException as error:
#     # print(error)
#     print(
#         "GDAL not found in environment. Waterdetect can still run as API calling DWImageClustering and passing"
#         " the arrays as dictionary. Refer to online documentation. No call to DWWaterDetect, that requires "
#         "loading satellite images from disk will be possible"
#     )
#     gdal = None

# Correct the jaccard score name depending on the sklearn version


# from .WaterDetect import DWWaterDetect
# from .Image import DWImageClustering
# from .Common import DWutils, DWConfig
# from .External_mask import prepare_external_masks

# class DWConfig:
#     pass
