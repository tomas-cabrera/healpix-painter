# Plotting utilities

import ligo.skymap.moc as lsm_moc
import matplotlib.pyplot as plt
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


def plot_skymap_gradient(ax, skymap_path, contours=[50, 90]):
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
    ax.imshow(img, cmap="cylon")

    return None


def plot_skymap_contours(ax, skymap_path, contours=[50, 90]):
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
                label=f"{cv}",
            )
    plt.legend()
    return None
