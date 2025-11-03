# Plotting utilities

import matplotlib.pyplot as plt

from .healpix import calc_contours_for_skymap

# Dictionary mapping filters to colors for plotting
FILTER2COLOR = {
    "u": "xkcd:indigo",
    "g": "xkcd:bluegreen",
    "r": "xkcd:orangered",
    "i": "xkcd:crimson",
    "z": "xkcd:black",
    "Y": "xkcd:gray",
}


def plot_skymap_with_contours(skymap_flat, contours, ax=None):
    # Initialize default axes if needed
    if ax is None:
        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
    # Plot skymap
    # ax.imshow_hpx(skymap_flat, cmap="cylon")
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
    plt.legend
    return ax
