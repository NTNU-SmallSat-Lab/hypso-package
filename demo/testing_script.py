# %% for dev, remove later
# Development use only:
import sys
# sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')
sys.path.insert(0, '/home/ariaa/smallSatLab/hypso-package-new/hypso/')

from hypso import Hypso1, Hypso2
import os
import matplotlib.pyplot as plt

from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file

# %% definitions
# dir_path = '/home/cameron/Dokumenter/Data/'
dir_path = '/home/ariaa/smallSatLab/data/'

# HYPSO-1 Capture
# h1_l1a_nc_file = os.path.join(dir_path, 'mvco_2025-01-13T14-57-34Z-l1a.nc')
# h1_points_file = os.path.join(dir_path, 'mvco_2025-01-13T14-57-34Z-l1a.points')
h1_core_path = os.path.join(dir_path, 'h1/lacrau_2024-12-26T10-24-27Z/lacrau_2024-12-26T10-24-27Z')

# HYPSO-2 Capture
h2_core_path = os.path.join(dir_path, 'h2/lacrau_2024-12-26T11-15-54Z/lacrau_2024-12-26T11-15-54Z')

level = 'l1c' # what processing level to start from

# %% Load HYPSO-1 Capture
# Load HYPSO-1 Capture
if level == 'l1a':
    # Generate L1b TOA radiance product
    satobj_h1 = Hypso1(path=h1_core_path + '-l1a.nc', verbose=True)
    satobj_h1.generate_l1b_cube()
    write_l1b_nc_file(satobj=satobj_h1, overwrite=True)
if level in ['l1a','l1b']:    
    # Generate L1c geolocated TOA radiance product
    satobj_h1 = Hypso1(path=h1_core_path + '-l1b.nc', verbose=True)
    satobj_h1.generate_l1c_cube()
    write_l1c_nc_file(satobj=satobj_h1, overwrite=True)
if level in ['l1a', 'l1b', 'l1c']:
    # Generate L1d TOA reflectance product
    satobj_h1 = Hypso1(path=h1_core_path + '-l1c.nc', verbose=True)
    satobj_h1.generate_l1d_cube()
    write_l1d_nc_file(satobj=satobj_h1, overwrite=True)
satobj_h1_original = Hypso1(path=h1_core_path + '-l1d_original.nc', verbose=True)
satobj_h1 = Hypso1(path=h1_core_path + '-l1d.nc', verbose=True)

# %% Load HYPSO-2 Capture
# Load HYPSO-1 Capture
if level == 'l1a':
    # Generate L1b TOA radiance product
    satobj_h2 = Hypso2(path=h2_core_path + '-l1a.nc', verbose=True)
    satobj_h2.generate_l1b_cube()
    write_l1b_nc_file(satobj=satobj_h2, overwrite=True)
if level in ['l1a','l1b']:    
    # Generate L1c geolocated TOA radiance product
    satobj_h2 = Hypso2(path=h2_core_path + '-l1b.nc', verbose=True)
    satobj_h2.generate_l1c_cube()
    write_l1c_nc_file(satobj=satobj_h2, overwrite=True)
if level in ['l1a', 'l1b', 'l1c']:
    # Generate L1d TOA reflectance product
    satobj_h2 = Hypso2(path=h2_core_path + '-l1c.nc', verbose=True)
    satobj_h2.generate_l1d_cube()
    write_l1d_nc_file(satobj=satobj_h2, overwrite=True)
satobj_h2_original = Hypso2(path=h2_core_path + '-l1d_original.nc', verbose=True)
satobj_h2 = Hypso2(path=h2_core_path + '-l1d.nc', verbose=True)


# %% Access datacubes
# Access datacubes
if level == 'l1a':
    l1a_cube = satobj_h1.l1a_cube
elif level in ['l1a', 'l1b']:
    l1b_cube = satobj_h1.l1b_cube
elif level in ['l1a', 'l1b', 'l1c']:
    l1c_cube = satobj_h1.l1c_cube
l1d_cube_original = satobj_h1_original.l1d_cube
l1d_cube = satobj_h1.l1d_cube

# %% plot origianal vs new spectral compensation (for testing difference in toa reflectance)
plt.figure()
plt.plot(l1d_cube_original[500,600,:], label='Original reflectance')
plt.plot(l1d_cube[500,600,:], label='New reflectance')
plt.legend()
plt.show()

# %% plot all three cubes for a single band
band = 40

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

# Plot data on each subplot
if level == 'l1a':
    axs[0, 0].imshow(l1a_cube[:,:,band])
    axs[0, 0].set_title('L1a band ' + str(band))
elif level in ['l1a', 'l1b']:
    axs[0, 1].imshow(l1b_cube[:,:,band])
    axs[0, 1].set_title('L1b band ' + str(band))
elif level in ['l1a', 'l1b', 'l1c']:
    axs[1, 0].imshow(l1c_cube[:,:,band])
    axs[1, 0].set_title('L1c band ' + str(band))

axs[1, 1].imshow(l1d_cube[:,:,band])
axs[1, 1].set_title('L1d band ' + str(band))

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
