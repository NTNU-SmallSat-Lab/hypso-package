
#import sys
#sys.path.append('../')
import georeferencing as georeferencing
import os





filename = '/home/cameron/Nedlastinger/Frohavet/frohavet_2024-04-15_1006Z-bin3-original.points'
gcps = georeferencing.GCPList(filename, origin_mode='qgis', cube_width=598, cube_height=1092)

print(gcps.image_mode)

gcps.change_image_mode(dst_image_mode='standard')

print(gcps.image_mode)


gcps.change_origin_mode(dst_origin_mode='cube')

print(gcps.origin_mode)

gcps.save()


