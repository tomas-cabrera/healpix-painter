Quickstart
==========

.. _installation:

Installation
------------

To use ``healpix-painter``, first install it using pip:

.. code-block:: console

   $ pip install healpix-painter

Basic Usage
-----------

The simplest way to use ``healpix-painter`` is through the command line:

.. code-block:: console

   $ healpix-painter --skymap-filename /path/to/skymap_dir/skymap.fits.gz
   Loading skymap...
   Calculating 90% contour regions...
   Loading DECam archival pointings...
   Fetching DECam archival pointings from https://astroarchive.noirlab.edu...
   Selecting pointings near 90% contour regions...
   Found 932 clustered pointings near 90% contour regions:
   	u: 52 pointings
   	g: 585 pointings
   	r: 489 pointings
   	i: 435 pointings
   	z: 491 pointings
   	Y: 25 pointings
   Evaluating healpix coverage of pointings with footprint...
   Selecting obsplan by 'probadd' scoring...
   Saving results in /path/to/skymap_dir...
   ========================================
   Coverage summary:
      u: 6.94% (15 pointings)
	   g: 92.09% (205 pointings)
	   r: 92.07% (203 pointings)
	   i: 92.43% (199 pointings)
	   z: 92.16% (194 pointings)
	   Y: 3.88% (6 pointings)
   ========================================

``healpix-painter`` writes several files to disc; the default output directory is the same as the directory containing the input skymap; the output directory can be specified with the ``--output-dir`` flag.

The main output files are filter-delimited CSV files containing the selected pointings and the amount of probabilty each pointing adds to the observation plan coverage (``pointings_<filter>.csv``).
The pointings are ordered in the file by the scoring metric used to select them; by default this means that each subsequent pointing is the one that adds the most probability to the total cumulative coverage.

``healpix-painter`` also generates several diagnostic plots.
The first of these plots the cumulative probability covered as a function of the number of pointings in the observation plan (``cumprob_npointings.png``).
This plot is intended to give a sense of the scale of resources required to cover the skymap:

.. image:: /_static/quickstart/cumprob_npointings.png
   :alt: Cumulative probability vs number of pointings plot
   :align: center
   :width: 80%

In addtition, filter-delimited plots of the skymap with the available coverage of the 90% area are generated (``footprints_<filter>_astro_hours_mollweide.png``):

.. image:: /_static/quickstart/footprints_g_astro_hours_mollweide.png
   :alt: All-sky skymap plot with available coverage
   :align: center
   :width: 80%

If the pointings cover a small enough area, zoomed-in plots are also generated (``footprints_<filter>_astro_hours_zoom.png``):

.. image:: /_static/quickstart/footprints_g_astro_hours_zoom.png
   :alt: Zoomed-in skymap plot with available coverage
   :align: center
   :width: 80%