Pixel windows
=============

``Product.window(latitude, longitude, size)`` is ``spectrum()`` for matchups:
the ``size`` x ``size`` pixels around the pixel nearest a site, and their
statistics.

.. code-block:: python

   w = l2a.window(45.3139, 12.5083, size=3)
   w["mean"], w["median"], w["std"], w["cv"]     # per band; cv = std / median
   w["pixels"]                                   # the window itself
   w.attrs["pixel_valid"], w.attrs["cv_median"], w.attrs["center_distance_m"]

   # leave flagged pixels out
   l2a.window(45.3139, 12.5083, size=5, exclude_flags=["land", "cloud"])

The window is clipped at the image edges (``box_size`` says what was used).
Statistics ignore NaN per band. A pixel counts as valid when any band in
``cv_band_range`` (default 405-570 nm) is finite, and ``cv_median`` is the
median CV over that range. ``mean_solar_zenith`` and the other angle
attributes are the window's means. No outlier rejection is applied:
following the usual matchup protocol (Bailey and Werdell, 2006), the
heterogeneity of the window shows in its CV instead.

These are the statistics hypso-processing-pipeline computes for HYPSO
matchups. The rest of a matchup (in-situ data, time windows, other sensors,
quality thresholds) belongs to the pipeline, not to this package.
