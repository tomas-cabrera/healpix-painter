# Plotting utilities

import ligo.skymap.moc as lsm_moc
import numpy as np
import reproject
from ligo.skymap import bayestar
from ligo.skymap.plot import cut_prime_meridian

from healpix_painter.healpix import calc_credible_levels_for_skymap, parse_skymap_args

# Dictionary mapping filters to colors for plotting
FILTER2COLOR = {
    "u": "#4477AA",
    "g": "#228833",
    "r": "#CCBB44",
    "i": "#EE6677",
    "z": "#AA3377",
    "Y": "#BBBBBB",
}


def plot_skymap_gradient(ax, skymap_path, imshow_kwargs={"cmap": "cylon"}):
    """Plots the skymap as a gradient on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the skymap.
    skymap_path : str
        The path to the skymap file.
    imshow_kwargs : dict, optional
        Keyword arguments to pass to matplotlib.pyplot.imshow, by default {"cmap": "cylon"}.

    Returns
    -------
    aximg : matplotlib.image.AxesImage
        The image object of the plotted skymap.
    """

    # Load skymap
    skymap = parse_skymap_args(skymap_filename=skymap_path)[1]
    if "UNIQ" in skymap.columns:
        skymap_flat = bayestar.rasterize(
            skymap,
            order=np.max(lsm_moc.uniq2order(skymap["UNIQ"])),
        )
    else:
        skymap_flat = skymap

    # Plot skymap
    img, mask = reproject.reproject_from_healpix(
        (skymap_flat["PROB"], "icrs"),
        ax.header,
        nested=True,
    )
    img = np.ma.masked_array(img, mask=~mask.astype(bool))
    aximg = ax.imshow(img, **imshow_kwargs)

    return aximg


def plot_skymap_contours(
    ax,
    skymap_path,
    contours=[50, 90],
    filled=False,
    contour_kwargs={
        "colors": "xkcd:bluegreen",
        "alpha": 0.8,
    },
):
    """Plots the specified contours on the skymap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the contours.
    skymap_path : str
        The path to the skymap file.
    contours : list, optional, default [50, 90]
        The contour levels to plot.
    filled : bool, optional, default False
        Whether to fill the contours.
    contour_kwargs : dict, optional
        Keyword arguments to pass to matplotlib.pyplot.contour.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The contour set of the plotted skymap.
    """

    # Load skymap
    skymap = parse_skymap_args(skymap_filename=skymap_path)[1]
    if "UNIQ" in skymap.columns:
        skymap_flat = bayestar.rasterize(
            skymap,
            order=np.max(lsm_moc.uniq2order(skymap["UNIQ"])),
        )
    else:
        skymap_flat = skymap

    # Calculate credible levels
    cls = calc_credible_levels_for_skymap(skymap_flat)
    # Plot contours
    if filled:
        contour_set = ax.contourf_hpx(
            cls, nested=True, levels=contours, **contour_kwargs
        )
    else:
        contour_set = ax.contour_hpx(
            cls, nested=True, levels=contours, **contour_kwargs
        )

    return contour_set


def plot_footprints(
    ax,
    footprint,
    scs,
    fill_kwargs={
        "color": "xkcd:bluegreen",
        "ls": "",
        "alpha": 0.5,
    },
):
    """Plot the given footprint at the given coordinates on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the footprint.
    footprint : healpix_painter.footprints.Footprint
        The footprint to plot.
    scs : astropy.coordinates.SkyCoord or list of astropy.coordinates.SkyCoord
        The coordinates at which to plot the footprint.
    fill_kwargs : dict, optional
        Keyword arguments to pass to matplotlib.pyplot.fill.

    Returns
    -------
    fills : list of matplotlib.patches.Polygon
        The list of filled polygons representing the plotted footprints.
    """

    # Ensure scs is iterable
    if scs.isscalar:
        scs = [scs]

    # Iterate over skycoords
    fills = []
    for sc in scs:
        _region_coords = footprint.rotate(sc.ra.deg, sc.dec.deg)
        # Iterate over CCDs:
        _regions = footprint.regions_from_region_coords(region_coords=_region_coords)
        for _region in _regions:
            # Get region vertices
            vertices = np.column_stack(
                [
                    np.deg2rad(_region.vertices.ra.deg),
                    np.deg2rad(_region.vertices.dec.deg),
                ]
            )
            # Divide over prime meridian and iterate
            for sub_vertices in cut_prime_meridian(vertices):
                fill = ax.fill(
                    np.rad2deg(sub_vertices[:, 0]),
                    np.rad2deg(sub_vertices[:, 1]),
                    transform=ax.get_transform("world"),
                    **fill_kwargs,
                )
                fills.append(fill)

    return fills
