
#import sys
#sys.path.append('../')
import georeferencing
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
points_file = 'erie_2022_08_27T16_05_36-bin3.points'

filename = os.path.join(dir_path, points_file)

height = 956
width = 228*3

gr = georeferencing.Georeferencer(filename, height, width)

print(gr.latitudes)
print(gr.longitudes)

