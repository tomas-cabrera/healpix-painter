# Plotting utilities

import ligo.skymap.moc as lsm_moc
import numpy as np
import reproject
from ligo.skymap import bayestar
from ligo.skymap.plot import cut_prime_meridian

from healpix_painter.healpix import (
    calc_credible_levels_for_skymap,
    parse_skymap_args,
)

# Dictionary mapping filters to colors for plotting
FILTER2COLOR = {
    "u": "xkcd:indigo",
    "g": "xkcd:bluegreen",
    "r": "xkcd:orangered",
    "i": "xkcd:crimson",
    "z": "xkcd:black",
    "Y": "xkcd:gray",
}


def plot_skymap_gradient(ax, skymap_path, imshow_kwargs={"cmap": "cylon"}):
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
    ax.imshow(img, **imshow_kwargs)

    return None


def plot_skymap_contours(
    ax,
    skymap_path,
    contours=[50, 90],
    plot_kwargs={
        "colors": "xkcd:bluegreen",
        "alpha": 0.8,
    },
):
    # Load skymap
    skymap = parse_skymap_args(skymap_filename=skymap_path)[1]
    if "UNIQ" in skymap.columns:
        skymap_flat = bayestar.rasterize(
            skymap,
            order=np.max(lsm_moc.uniq2order(skymap["UNIQ"])),
        )
    else:
        skymap_flat = skymap

    # Plot contours by contouring the credible levels on the reprojected
    # pixel grid; this avoids spurious lines from paths that wrap across
    # the map's RA discontinuity, which occur when contouring in world
    # coordinates directly.
    cls = calc_credible_levels_for_skymap(skymap_flat)
    ax.contour_hpx(cls, nested=True, levels=contours, **plot_kwargs)
    return None


def plot_footprints(
    ax,
    footprint,
    scs,
    plot_kwargs={
        "color": "xkcd:bluegreen",
        "ls": "",
        "alpha": 0.5,
    },
):
    # Iterate over skycoords
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
                ax.fill(
                    np.rad2deg([*sub_vertices[:, 0], sub_vertices[0, 0]]),
                    np.rad2deg([*sub_vertices[:, 1], sub_vertices[0, 1]]),
                    transform=ax.get_transform("world"),
                    **plot_kwargs,
                )
