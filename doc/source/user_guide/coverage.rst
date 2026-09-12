.. _coverage:

Calculating HEALPix Skymap Coverage
===================================

This section details how to calculate the coverage of a HEALPix skymap by a given observation plan, using :func:`healpix_painter.healpix.calc_skymap_coverage`.

Single-Facility Coverage
------------------------

.. ipython:: python

    import os.path as pa

    DATA_DIR = "doc/data"

    # Select a sample skymap
    skymap_path = pa.join(DATA_DIR, "S240511i.Bilby.multiorder.fits")

The skymap path is directly passed to :func:`calc_skymap_coverage() <healpix_painter.healpix.calc_skymap_coverage>`;
the pointings must be passed as an `Astropy SkyCoord object <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html>`__.

.. ipython:: python

    # Prepare pointings
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    pointings_path = pa.join(DATA_DIR, "S240511i.pointings_decam.csv")

    pointings_table = Table.read(pointings_path, format="csv")

    pointings_skycoord = SkyCoord(ra=pointings_table['ra'], dec=pointings_table['dec'], unit='deg')
    
    pointings_skycoord

After selecting a :class:`healpix_painter.footprints.Footprint` object, the coverage can be calculated.

.. ipython:: python

    from healpix_painter.telescopes.decam import DECamFootprint
    from healpix_painter.healpix import calc_skymap_coverage

    calc_skymap_coverage(skymap_path, pointings_skycoord, DECamFootprint)

Multi-Facility Coverage
------------------------

:func:`calc_skymap_coverage() <healpix_painter.healpix.calc_skymap_coverage>` can also calculate the coverage of a skymap by an observation plan involving multiple facilities.
This can be done by passing lists to the ``pointings_list`` and ``footprints_list`` arguments, where ``pointings_list[i]`` is the SkyCoord object containing the pointings for the footprint ``footprints_list[i]``.

.. ipython:: python

    from healpix_painter.telescopes.ztf import ZTFFootprint

    pointings_skycoord_ztf = pointings_skycoord[0].copy()

    calc_skymap_coverage(
        skymap_path,
        [pointings_skycoord, pointings_skycoord_ztf],
        [DECamFootprint, ZTFFootprint],
    )

Here, the first element of the output tuple is the total probability covered by all telescopes;
the second element is a list of the probabilities covered by each telescope, in the same order as the input lists.