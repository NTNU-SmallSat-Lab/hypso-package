Coefficient history
===================

How the calibration coefficient files came about, for the record. Which file
each correction uses today, by version, is in each calibration package's
``data/coefficients.yaml`` (:doc:`calibration`).

HYPSO-1
-------

The calibration team's version log, verbatim (it describes the HYPSO-1 v0 and
v1 coefficients; file names are the CSV names of the time, now shipped as
``.npz`` with the same base names):

.. code-block:: text

   #### HYPSO-1 calibration coefficient version descriptions ####

   ---------------------------------------

   v0 (before version numbers were used): 
   May 2023

   Spectral calibration:
   - Filename: spectral_bands_HYPSO-1_120bands.csv 
   - Center wavelength of each band

   Radiometric calibration:
   - Either from 10 img avg cropped and binned (only nominal) calibration data or from single full calibration frame
   - With or without destriping
   - No spatial shift applied

   ---------------------------------------

   v1:
   September 2023

   Spectral calibration:
   - Filename from script: spectral_bands_HYPSO-1_120bands.csv (same as v0)
   - Filename used: spectral_bands_HYPSO-1_v1.csv 
   - Center wavelength of each band

   Radiometric calibration:
   - Nominal capture (684 pixels x 120 bands)
       - Filename from script: radiometric_calibration_matrix_HYPSO-1_shifted_nominal.csv
       - Filename used: radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv
       - Created using average of 10 cropped to AOI (nominal) calibration frames
       - Background level: 8 counts
       - Used spectral calibration matrix (same as used for smile correction) to determine correct wavelength for each pixel - to give correct radiance value regardless of smile
       - Applied 7 pixel shift to correct for spatial shift (padding with zeros)
       - Used data from single full frame radiometric calibration data (crop and bin) to exchange 0s resulting from pixel shift with non-averaged radiometric calibration values
       - No destriping (is a separate matrix)
   - Wide capture (1092 pixels x 120 bands)
       - Filename from script: radiometric_calibration_matrix_HYPSO-1_shifted_wide.csv
       - Filename used: radiometric_calibration_matrix_HYPSO-1_wide_v1.csv
       - Created from the single full frame radiometric calibration data
       - Background level: 8 counts
       - Used spectral calibration matrix (same as used for smile correction) to determine correct wavelength for each pixel - to give correct radiance value regardless of smile
       - Applied 7 pixel shift to correct for spatial shift (padding with zeros)
       - Cropped and binned using params:
           -     row_count = 1092
           -     column_count = 1080
           -     bin_factor = 9
           -     aoi_x = 428
           -     aoi_y = 62
       - No destriping (is a separate matrix)


   Smile correction:
   - Nominal capture (684 pixels x 120 bands)
       - Filename from script: spectral_calibration_matrix_HYPSO-1_shifted_nominal.csv
       - Filename used: smile_correction_matrix_HYPSO-1_nominal_v1.csv
       - Used spectral calibration data from full frame
       - Applied 7 pixel shift to correct for spatial shift (padding with zeros)
       - Cropped and binned using params:
           -     row_count = 684
           -     column_count = 1080
           -     bin_factor = 9
           -     aoi_x = 428
           -     aoi_y = 266
   - Wide capture (1092 pixels x 120 bands)
       - Filename from script: spectral_calibration_matrix_HYPSO-1_shifted_wide.csv
       - Filename used: smile_correction_matrix_HYPSO-1_wide_v1.csv
       - Used spectral calibration data from full frame
       - Applied 7 pixel shift to correct for spatial shift (padding with zeros)
       - Cropped and binned using params:
           -     row_count = 1092
           -     column_count = 1080
           -     bin_factor = 9
           -     aoi_x = 428
           -     aoi_y = 62


   Destriping correction:
   - Nominal capture (684 pixels x 120 bands)
       - Filename from script: n/a
       - Filename used: destriping_matrix_HYPSO-1_nominal_v1
       - based off lacrau_2023-09-09
       - compute along-track derivative after log transform over water pixels, for multiplicative correction
   - Wide capture (1092 pixels x 120 bands)
       - Filename from script: n/a
       - Filename used: destriping_matrix_HYPSO-1_wide_v1
       - based off finnmark_2022-08-06_1035Z


   ---------------------------------------

HYPSO-2
-------

``hypso2_calibration``'s ``data/coefficients.yaml`` records, for each file,
what it is and its name before the V1-V3 releases (the ``was ...`` notes).

Smile and keystone model
------------------------

``hypso2_calibration`` also ships ``h2_distortion_model_v1.npz``: the
coefficients ``A_hat`` and ``B_hat`` (previously the files ``H2_A-hat`` and
``H2_B-hat``) of HYPSO-2's detector distortion model after Lawrence (2003), as
used in the calibration team's processing. For each detector pixel (``x``
spectral, ``y`` spatial) the true position is a second-order polynomial with the
terms ``[1, x, y, xy, x², y²]``: ``A_hat`` gives the spectral shift (smile),
``B_hat`` the spatial shift (keystone, -2.9 to +1.8 pixels across the full
frame). The shipped V2 smile matrix agrees with the ``A_hat`` model within about
2 nm, so it was probably derived from it. The keystone correction is not applied
by default; it is available for review (``keystone`` in ``coefficients.yaml``,
disabled; ``calibrate(keystone=True)`` applies it, with a warning).

Removed source files
--------------------

Until 2026-10-27 the repository also held the coefficients as CSV files (a
top-level ``coefficients/`` folder, about 400 MB) with a script converting them
to ``.npz``. Nothing read them, and the ``.npz`` files hold the same values
(checked, NaN included). They are in the git history before that date.
