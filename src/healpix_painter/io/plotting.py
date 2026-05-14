# Plotting utilities

import ligo.skymap.moc as lsm_moc
import numpy as np
import reproject
from ligo.skymap import bayestar

from healpix_painter.healpix import calc_contours_for_skymap, parse_skymap_args

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


def plot_skymap_contours(ax, skymap_path, contours=[50, 90], plot_kwargs={}):
    # Load skymap
    skymap = parse_skymap_args(skymap_filename=skymap_path)[1]
    if "UNIQ" in skymap.columns:
        skymap_flat = bayestar.rasterize(
            skymap,
            order=np.max(lsm_moc.uniq2order(skymap["UNIQ"])),
        )
    else:
        skymap_flat = skymap

    # Plot contours
    cs = calc_contours_for_skymap(skymap_flat, contours)
    for c, cv in zip(cs, contours):
        for m in c:
            ax.plot(
                [v[0] for v in m],
                [v[1] for v in m],
                transform=ax.get_transform("world"),
                **plot_kwargs,
            )
    return None


def plot_footprints(ax, footprint, scs, plot_kwargs={}):
    # Iterate over skycoords
    for sc in scs:
        _region_coords = footprint.rotate(sc.ra.deg, sc.dec.deg)
        # Iterate over CCDs:
        for _ccd in _region_coords:
            _x = _ccd[0, :]
            _y = _ccd[1, :]
            _x = np.append(_x, _x[0])
            _y = np.append(_y, _y[0])
            ax.plot(
                _x,
                _y,
                transform=ax.get_transform("world"),
                **plot_kwargs,
            )
        # NB: This uses the plotting provided by astropy.regions; slower, but may be better in the future
        #       Presently does not handle the wrapping at the edges of Mollweide well
        # _regions = footprint.regions_from_region_coords(region_coords=_region_coords)
        # for _region in _regions:
        #     _pixel_region = _region.to_pixel(WCS(ax.header))
        #     _pixel_region.plot(ax=ax, **plot_kwargs)
