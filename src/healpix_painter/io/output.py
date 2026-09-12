import os
import os.path as pa
import shutil

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from healpix_painter.io.plotting import (
    FILTER2COLOR,
    plot_footprints,
    plot_skymap_gradient,
)


def skycoord_median(coords: SkyCoord) -> SkyCoord:
    """
    Compute a median of a set of SkyCoords by averaging
    their 3D unit vectors.

    Parameters
    ----------
    coords : SkyCoord
        A SkyCoord with one or more positions.

    Returns
    -------
    SkyCoord
        A median position in ICRS, in degrees.
    """
    # Return if single coordinate
    if coords.isscalar:
        return coords
    if len(coords) == 1:
        return coords[0]

    # Convert to ICRS Cartesian unit vectors
    cart = coords.icrs.cartesian
    x, y, z = cart.x.value, cart.y.value, cart.z.value

    # Mean of unit vectors
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

    # Re-normalize (projects back onto unit sphere)
    norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    x_c, y_c, z_c = x_mean / norm, y_mean / norm, z_mean / norm
    # Convert back to spherical (RA/Dec)
    ra_c = np.degrees(np.arctan2(y_c, x_c)) % 360.0
    dec_c = np.degrees(np.arcsin(z_c))

    return SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg, frame="icrs")


# Output results
def package_results(
    skymap_filename,
    selected_pointings,
    footprint,
    output_dir=None,
):
    """Format painter output to standarized files and plots.

    Parameters
    ----------
    skymap_filename : str
        The path to the skymap used for tiling.
    selected_pointings : dict of pd.DataFrames
        Dictionary containing the selected pointings.
        Keys should be filter names, and values should be pandas DataFrames with columns 'ra', 'dec', and 'probs_added', with the ultimate column containing the amount of probability that pointing adds to the observation plan.
    footprint : healpix_painter.footprints.Footprint
        The footprint object used for tiling.
    output_dir : str, optional
        The directory to save the output in.
        If None, uses the directory the skymap is in (default behavior).
    """
    # Copy skymap to output directory, or define output as skymap directory
    if output_dir is not None:
        # Make output dir
        os.makedirs(output_dir, exist_ok=True)
        # Copy skymap
        shutil.copy(skymap_filename, output_dir)
    else:
        output_dir = pa.dirname(skymap_filename)
    print(f"Saving results in {output_dir}...")

    # Make prob vs n_pointings plot
    plt.figure(figsize=(4, 3))
    print("=" * 40)
    print("Coverage summary:")
    for f in selected_pointings:
        n_pointings = selected_pointings[f].shape[0]
        total_prob = np.sum(selected_pointings[f]["probs_added"])
        print(f"\t{f}: {total_prob * 100:.2f}% ({n_pointings} pointings)")
        plt.plot(
            np.arange(n_pointings) + 1,
            np.cumsum(selected_pointings[f]["probs_added"]),
            label=f"{f} {(np.sum(selected_pointings[f]['probs_added']) * 100):.2f}% covered",
            color=FILTER2COLOR.get(f),
        )
    print("=" * 40)
    plt.legend()
    plt.xlabel("Number of Pointings")
    plt.ylabel("Cumulative Probability Covered")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(pa.join(output_dir, "cumprob_npointings.png"))
    plt.savefig(pa.join(output_dir, "cumprob_npointings.pdf"))
    plt.close()

    # Save footprint plots and csvs
    # Set pointing limit for plotting
    # TODO: change pointing limit, currently 480 (about 12 hours of 60s DECam exposures)
    N_EXP_MAX = 480
    # Set mollweide kwargs
    axes_kwargs = [
        {
            "projection": "astro hours mollweide",
        }
    ]
    # Check whether pointings are all in one hemisphere; if so, add zoomed-in plot
    # Collect all pointings into a single SkyCoord
    ras = []
    decs = []
    for f in selected_pointings:
        ras.extend(selected_pointings[f]["ra"].values)
        decs.extend(selected_pointings[f]["dec"].values)
    scs = SkyCoord(
        ra=ras * u.deg,
        dec=decs * u.deg,
        frame="icrs",
    )
    sc_pointings_median = skycoord_median(scs)
    sc_separations = scs.separation(sc_pointings_median)
    max_radius = np.max(sc_separations) * 1.1  # Add 10% margin
    if max_radius > 90 * u.deg:
        print("Pointings span more than one hemisphere; skipping zoom plot.")
    else:
        axes_kwargs.append(
            {
                "projection": "astro hours zoom",
                "center": sc_pointings_median.to_string("hmsdms"),
                "radius": f"{max_radius.to_string(decimal=True)} deg",
            }
        )
    # Iterate over filters
    for f in selected_pointings:
        # Get + save data
        filter_data = selected_pointings[f]
        filter_data.to_csv(pa.join(output_dir, f"pointings_{f}.csv"), index=False)
        # Set pointing limit for plotting
        n_exp = min(N_EXP_MAX, len(filter_data["probs_added"]))
        # Get SkyCoords for plotting
        filter_scs = SkyCoord(
            ra=filter_data["ra"].values[:n_exp] * u.deg,
            dec=filter_data["dec"].values[:n_exp] * u.deg,
            frame="icrs",
        )
        # Plot footprints
        for ak in axes_kwargs:
            # Initialize axes
            ax = plt.axes(**ak)
            ax.grid()
            # Plot skymap
            plot_skymap_gradient(ax, skymap_filename)
            # Plot coverage
            plot_footprints(
                ax,
                footprint,
                filter_scs,
                fill_kwargs={
                    "color": FILTER2COLOR.get(f),
                    "ls": "",
                    "alpha": 0.5,
                },
            )
            # Save
            plt.savefig(
                pa.join(
                    output_dir,
                    f"footprints_{f}_{ak['projection'].replace(' ', '_')}.png",
                )
            )
            plt.savefig(
                pa.join(
                    output_dir,
                    f"footprints_{f}_{ak['projection'].replace(' ', '_')}.pdf",
                )
            )
            plt.close()
