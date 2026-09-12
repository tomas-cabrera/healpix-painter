.. _plotting:

Plotting Utilities
==================

This section describes the plotting ultilities available in ``healpix-painter``.
In general, these functions facilitate the addition of some element to an existing Matplotlib Axes,
and are best used with Axes created with `the special WCSAxes subclasses provided by ligo.skymap <https://lscsoft.docs.ligo.org/ligo.skymap/plot/allsky.html>`__.

Plotting Skymaps as Gradients
-----------------------------

After initializing a suitable Axes object,
skymaps can be plotted as gradients using :func:`healpix_painter.io.plotting.plot_skymap_gradient`:

.. ipython:: python

    import os.path as pa
    import matplotlib.pyplot as plt
    import ligo.skymap.plot
    from healpix_painter.io.plotting import plot_skymap_gradient

    DATA_DIR = "doc/data"

    # Initialize axes
    ax = plt.axes(projection="astro hours mollweide")

    # Add grid lines
    ax.grid()

    # Select a sample skymap
    skymap_path = pa.join(DATA_DIR, "S240511i.Bilby.multiorder.fits")

    # Plot skymap
    @savefig gradient_mollweide.png width=4in
    plot_skymap_gradient(ax, skymap_path)

Other ligo.skymap Axes subclasses can be used,
and arguments for the underlying :func:`matplotlib.pyplot.imshow` call can be passed:

.. ipython:: python

    ax = plt.axes(
        projection="astro hours zoom",
        center="11.25h -19d",
        radius="11 deg",
    )

    ax.grid()

    # Plot skymap
    @savefig gradient_zoom.png width=4in
    plot_skymap_gradient(
        ax,
        skymap_path,
        imshow_kwargs={
            "cmap": "plasma",
            "alpha": 0.8,
        },
    )

Plotting Skymaps as Contours
----------------------------

Plotting skymap contours functions similarly to plotting gradients:

.. ipython:: python

    from healpix_painter.io.plotting import plot_skymap_contours

    ax = plt.axes(
        projection="astro hours zoom",
        center="11.25h -19d",
        radius="11 deg",
    )

    ax.grid()

    # Plot contours
    @savefig contours_basic.png width=4in
    plot_skymap_contours(ax, skymap_path)

This function calls the ligo.skymap `contour_hpx <https://lscsoft.docs.ligo.org/ligo.skymap/plot/allsky.html#ligo.skymap.plot.allsky.AutoScaledWCSAxes.contour_hpx>` function by default;
setting ``filled=True`` will call `contourf_hpx <https://lscsoft.docs.ligo.org/ligo.skymap/plot/allsky.html#ligo.skymap.plot.allsky.AutoScaledWCSAxes.contourf_hpx>` instead, for filled contours.
Parameters for either can be set with the ``contour_kwargs`` argument:

.. ipython:: python

    from healpix_painter.io.plotting import plot_skymap_contours

    ax = plt.axes(
        projection="astro hours zoom",
        center="11.25h -19d",
        radius="11 deg",
    )

    ax.grid()

    plot_skymap_contours(
        ax,
        skymap_path,
        contours=[90],
        contour_kwargs={
            "colors": "xkcd:crimson",
            "linestyles": "dashed",
            "linewidths": 5,
            "alpha": 0.5,
        },
    )

    @savefig contours_extra.png width=4in
    plot_skymap_contours(
        ax,
        skymap_path,
        contours=[0,50],
        filled=True,
        contour_kwargs={
            "colors": "xkcd:darkblue",
            "alpha": 0.7,
        },
    )

Plotting Telescope Footprints
-----------------------------

To plot on-sky footprints, the pointings must first be cast into an `Astropy SkyCoord object <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html>`__:

.. ipython:: python

    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    pointings_table = Table.read(pa.join(DATA_DIR, "S240511i.pointings_decam.csv"))

    skycoord_decam = SkyCoord(
        ra=pointings_table["ra"],
        dec=pointings_table["dec"],
        unit="deg",
    )
    
    skycoord_ztf = SkyCoord(
        ra=pointings_table["ra"][0],
        dec=pointings_table["dec"][0],
        unit="deg",
    )

Footprints can then be added to an initialized Axes using :func:`healpix_painter.io.plotting.plot_footprints`,
after specifying the appropriate :class:`healpix_painter.footprints.Footprint` object:

.. ipython:: python

    from healpix_painter.io.plotting import plot_footprints
    from healpix_painter.telescopes.decam import DECamFootprint
    from healpix_painter.telescopes.ztf import ZTFFootprint

    ax = plt.axes(
        projection="astro hours zoom",
        center="11.25h -19d",
        radius="11 deg",
    )

    ax.grid()

    # DECam pointings
    plot_footprints(
        ax,
        DECamFootprint,
        skycoord_decam,
    );

    # ZTF pointings
    @savefig footprints.png width=4in
    plot_footprints(
        ax,
        ZTFFootprint,
        skycoord_ztf,
        fill_kwargs={
            "color": "xkcd:orange",
            "fill": False,
            "linewidth": 2,
        },
    );

See `matplotlib.axes.Axes.fill <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill.html>`__ for the available keywords to change the footprint appearance.

Footprint plotting behaves mostly as expected near the poles, but some artifacts may be present along the prime meridian:

.. ipython:: python

    pointings_poles = Table.read(pa.join(DATA_DIR, "S250830bp.pointings_decam.csv"))

    ax = plt.axes(
        projection="astro hours zoom",
        center="3.5h -88.5d",
        radius="2 deg",
    )

    ax.grid()

    skycoord_poles = SkyCoord(
        ra=[15, 75],
        dec=[-89, -88],
        unit="deg",
    )

    @savefig footprints_poles.png width=4in
    plot_footprints(
        ax,
        DECamFootprint,
        skycoord_poles,
    );