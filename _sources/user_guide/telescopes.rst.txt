.. _telescopes:

Telescope Implementation
========================

This section is still a work in progress, as the telescope data structure is still being generalized.
Below are some details on custom footprints may be implemented for use in coverage calculation and plotting.

Footprints
----------

``healpix_painter`` currently has footprints implemented for the DECam and ZTF telescopes.
If you are interested in implementing your own footprints, see the details below.
It may also be useful to examine the example scripts `get_footprint_coords_decam.py <https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/get_footprint_coords_decam.py>`__
and `get_footprint_coords_ztf.py <https://github.com/tomas-cabrera/healpix-painter/blob/main/examples/get_footprint_coords_ztf.py>`__, which were used to generate the respective files,

``healpix-painter`` interacts with telescope footprints through the :class:`healpix_painter.footprints.Footprint` class.
Within this class, footprint geometry is saved as a `regions.Regions <https://astropy-regions.readthedocs.io/en/stable/api/regions.Regions.html>`__ object,
consisting of a collection of `Region <https://astropy-regions.readthedocs.io/en/stable/api/regions.Region.html>`__ objects,
each representing a single CCD in the telescope focal plane.

A :func:`Footprint() <healpix_painter.footprints.Footprint>` object can be initialized with either

1. An existing `CASA Region Text Format (CRTF) <https://casadocs.readthedocs.io/en/stable/notebooks/image_analysis.html#Region-File-Format>`__ file, or
2. An ``n_ccd`` x 2 x ``n_vertices_per_ccd`` array of CCD coordinates (with the second axis iterating over RA and declination). The coordinates should be the (``RA``, ``dec``) of the vertices when the telescope is pointed at (``RA``, ``dec``) = (0, 0).

The creation of a new CRTF from a set of coordinates is facilitated in the :func:`healpix_painter.footprints.make_footprint_crtf` function.
This function can also generate a CRTF file for the convex hull of a footprint;
depending on the original footprint geometry, using a convex hull may significantly speed up certain operations.

.. admonition:: Current Limitations

    1. All CCDs in a footprint must have the same number of vertices, to avoid errors with how coordinate transformations are implemented.

    2. Only equatorial-mount telescopes are currently supported;
    support of mounts which rotate the focal plane (e.g. alt-azimuth) is still under development. 

