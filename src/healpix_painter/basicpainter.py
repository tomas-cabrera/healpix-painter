import os.path as pa

import astropy.units as u
import ligo.skymap.moc as lsm_moc
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky

from healpix_painter import healpix
from healpix_painter.footprints import DUMMY_WCS, DECamConvexHull
from healpix_painter.tilings import decam
from healpix_painter.tilings.clustering import cluster_skycoord


def basic_painter(
    healpixfilename=None,
    lvkeventid=None,
    footprint=DECamConvexHull,
    max_sep_cluster=15 * u.arcmin,
):
    # Load skymap
    sm = healpix.parse_skymap_args(healpixfilename, lvkeventid)
    # Calculate contour regions
    # Flatten skymap
    sm_flat = lsm_moc.rasterize(sm)
    # plot_skymap_with_contours(sm_flat, [50, 90])
    r90s = healpix.get_skymap_contours_as_regions(sm_flat, [90])[0]
    # Load pointings
    decam_tiling = decam.get_archival_tiling(force_update=False)
    # Get pointings within contours
    sc_tiling = SkyCoord(
        decam_tiling["ra"],
        decam_tiling["dec"],
        unit=u.deg,
    )
    # TODO: Back of the envelope calculation shows that 12 hours / (60+30)s * 3 sq. deg. = 1440 square degrees of coverage.
    #       Based on this, choosing the hpx area to get pointings for should cut off at ~1500 square degrees, or 95%, whichever is smaller
    in_region = [False] * decam_tiling.shape[0]
    for r90 in r90s:
        in_region_temp = r90.contains(sc_tiling, DUMMY_WCS)
        in_region = np.logical_or(in_region, in_region_temp)
    nearby_tiling = decam_tiling[in_region]
    print(nearby_tiling)
    del decam_tiling
    # Cluster pointings
    nearby_tiling_skycoord = SkyCoord(
        nearby_tiling["ra"],
        nearby_tiling["dec"],
        unit=u.deg,
    )
    nearby_tiling_clustered_skycoord = cluster_skycoord(
        nearby_tiling_skycoord,
        max_sep=max_sep_cluster,
    )
    # Determine coverage of clustered pointings
    nearby_coverage = {
        "ra": nearby_tiling_clustered_skycoord.ra.to(u.deg),
        "dec": nearby_tiling_clustered_skycoord.dec.to(u.deg),
    }
    # Iterate over filters
    for f in decam.FILTERS:
        # Select filters
        filter_skycoord = nearby_tiling_skycoord[nearby_tiling[f]]
        if len(filter_skycoord) == 0:
            nearby_coverage[f] = False
        else:
            # Perform crossmatch
            idx, d2d, _ = match_coordinates_sky(
                nearby_tiling_clustered_skycoord,
                filter_skycoord,
            )
            # Save coverage
            nearby_coverage[f] = d2d <= max_sep_cluster
    # Cast to dataframe + clean
    nearby_coverage = pd.DataFrame(nearby_coverage)
    del nearby_tiling, nearby_tiling_skycoord, nearby_tiling_clustered_skycoord
    # Determine coverage of healpixs
    # Ends up as an [n_pointings x n_healpixels] array, true if that pointing covers the healpixel
    # Get ra/dec for skymap
    sm["RA"], sm["DEC"] = healpix.calc_radecs_for_skymap(sm)
    sc_sm = SkyCoord(
        sm["RA"],
        sm["DEC"],
        unit="deg",
        frame="icrs",
    )
    # Iterate over exposures
    in_footprint = []
    for _, pointing in nearby_coverage.iterrows():
        # Find healpixs in footprint
        in_pointing = footprint.in_footprint(
            ra_obj=sc_sm.ra.to(u.deg).value,
            dec_obj=sc_sm.dec.to(u.deg).value,
            ra_exp=pointing["ra"],
            dec_exp=pointing["dec"],
        )
        # Append
        in_footprint.append(in_pointing)
    # Cast as array
    in_footprint = np.array(in_footprint)
    # Select pointings
    # Get probability contained in hpxs
    hpx_probs = healpix._get_probs_for_skymap(sm)
    # Iterate over filters
    # TODO: The algorithm works, but the output was sloppily put together right before bedtime, so definitely rethink that
    result = {}
    for f in decam.FILTERS:
        i_exps = []
        probs_added = []
        hpx_probs_uncovered = hpx_probs.copy()
        # Iterate until no more coverage is possible
        while True:
            # Calculate probability coverage using only non-covered pixels
            exp_probs = (in_footprint * hpx_probs_uncovered).sum(
                axis=1
            ) * nearby_coverage[f]
            # Break if no exposures cover new prob
            if exp_probs.max() == 0.0:
                break
            # Select exposure covering the most probability
            i_exp = np.argmax(exp_probs)
            # Append coverage to total coverage (as list, so gradual coverage can be plotted)
            i_exps.append(i_exp)
            probs_added.append(exp_probs[i_exp])
            # Mark healpixes as covered (set probability to 0)
            hpx_probs_uncovered[in_footprint[i_exp, :]] = 0.0
        # Save
        result[f] = {
            "i_exps": i_exps,
            "probs_added": probs_added,
        }
    import matplotlib.pyplot as plt

    f2c = {
        "u": "xkcd:blue",
        "g": "xkcd:bluegreen",
        "r": "xkcd:orangered",
        "i": "xkcd:crimson",
        "z": "xkcd:black",
        "Y": "xkcd:gray",
    }
    for f in decam.FILTERS:
        plt.plot(
            np.arange(len(result[f]["i_exps"])),
            np.cumsum(result[f]["probs_added"]),
            label=f"{f} {(np.sum(result[f]['probs_added']) * 100):.2f}% covered",
            color=f2c[f],
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(pa.join(pa.dirname(healpixfilename), "prob_npointings.png"))
    plt.savefig(pa.join(pa.dirname(healpixfilename), "prob_npointings.pdf"))
    plt.show()
    plt.close()
    for f in decam.FILTERS:
        n_exp = min(80, nearby_coverage[f].sum())
        if n_exp == 0:
            continue
        x = [nearby_coverage.iloc[i]["ra"] for i in result[f]["i_exps"]]
        y = [nearby_coverage.iloc[i]["dec"] for i in result[f]["i_exps"]]
        alpha = np.interp(
            result[f]["probs_added"],
            (np.min(result[f]["probs_added"]), np.max(result[f]["probs_added"])),
            (0.2, 1),
        )
        df = pd.DataFrame({"ra": x, "dec": y, "prob": result[f]["probs_added"]})
        df.to_csv(
            pa.join(pa.dirname(healpixfilename), f"pointings_{f}.csv"), index=False
        )
        lw = alpha + 1
        # plt.scatter(
        #     x,
        #     y,
        #     color=f2c[f],
        #     alpha=alpha,
        # )
        for i in np.arange(min(80, len(x))):
            fpc = footprint.rotate(x[i], y[i])
            for fpccd in fpc:
                plt.plot(
                    fpccd[0],
                    fpccd[1],
                    color=f2c[f],
                    alpha=alpha[i],
                    lw=lw[i],
                )
        plt.title(
            f"{f}, {(100 * np.sum(result[f]['probs_added'][: min(80, len(x))])):.2f}% covered"
        )
        # plt.legend()
        plt.tight_layout()
        plt.savefig(pa.join(pa.dirname(healpixfilename), f"pointings_{f}.png"))
        plt.savefig(pa.join(pa.dirname(healpixfilename), f"pointings_{f}.pdf"))
        plt.show()
        plt.close()
