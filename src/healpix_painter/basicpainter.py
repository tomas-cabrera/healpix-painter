import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from joblib import Parallel, delayed

from healpix_painter import healpix
from healpix_painter.io.output import package_results
from healpix_painter.telescopes import decam
from healpix_painter.telescopes.decam import DECamConvexHull
from healpix_painter.tilings.clustering import cluster_skycoord


def _pointing_in_footprint(ra_exp, dec_exp, ra_obj, dec_obj, footprint):
    """Wrapper of footprint.in_footprint; included for parallelization."""
    return footprint.in_footprint(
        ra_obj=ra_obj,
        dec_obj=dec_obj,
        ra_exp=ra_exp,
        dec_exp=dec_exp,
    )


def score_by_probadd(hpx_probs, in_footprint):
    """Return the total probability contained by each pointing.

    Parameters
    ----------
    hpx_probs : np.ndarray
        n_healpixels x 1 array of probabilities for each healpixel in the skymap.
    in_footprint : np.ndarray
        n_pointings x n_healpixels array indicating which healpixels are in each pointing's footprint.

    Returns
    -------
    np.ndarray
        n_pointings x 1 array of scores for each pointing.
    """

    # Calculate probability coverage using only non-covered pixels
    exp_probs = (in_footprint * hpx_probs).sum(axis=1)
    # Return
    return exp_probs


def score_by_probden_probadd(hpx_probs, in_footprint):
    """Return a score for each pointing based on the maximum probability density in each pointing, breaking ties by total probability added.

    Parameters
    ----------
    hpx_probs : np.ndarray
        n_healpixels x 1 array of probabilities for each healpixel in the skymap.
    in_footprint : np.ndarray
        n_pointings x n_healpixels array indicating which healpixels are in each pointing's

    Returns
    -------
    np.ndarray
        n_pointings x 1 array of scores for each pointing.
    """

    # Mask hpx not in footprints
    footprint_hpx_probs = in_footprint * hpx_probs
    # Get maximum prob
    max_prob = footprint_hpx_probs.max(axis=1)
    # Calculate probability coverage using only non-covered pixels
    probadd = footprint_hpx_probs.sum(axis=1)
    # Make dataframe
    df = pd.DataFrame({"max_prob": max_prob, "probadd": probadd})
    # Sort by max_prob, then probadd
    df.sort_values(
        ["max_prob", "probadd"],
        inplace=True,
    )
    # Add scores (zero out scores for events that cover no probability)
    df["scores"] = np.arange(df.shape[0]) * (df["probadd"] != 0)
    scores = df.sort_index()["scores"]
    # Return
    return scores


def basic_painter(
    skymap_filename=None,
    lvk_eventname=None,
    footprint=DECamConvexHull,
    tiling_force_update=False,
    max_sep_cluster=1.0 * u.arcmin,
    scoring="probadd",
    output_dir=None,
    n_jobs=-1,
):
    """Generate an observation plan for the skymap using the given footprint and tiling.
    At each step, chooses the highest scoring pointing, and then removes the healpixels covered by that pointing from the skymap before proceeding.

    Currently uses a tiling of archival DECam coverage from the NOIRLab AstroData Archive.

    Parameters
    ----------
    skymap_filename : str, optional
        The path to the HEALPix skymap to tile.
        Either skymap_filename or lvk_eventname must be provided.
    lvk_eventname : str, optional
        The LVK event id to tile.
        Either skymap_filename or lvk_eventname must be provided.
    footprint : healpix_painter.footprints.Footprint, optional
        The telescope footprint to use for tiling; DECamConvexHull by default.
    tiling_force_update : bool, optional
        Whether to force update the tiling cache; False by default.
    max_sep_cluster : astropy.coordinates.Angle, optional
        The radius to use when clustering pointings, by default 1 arcmin.
    scoring : {'probadd', 'probden_probadd'}, default 'probadd'
        The scoring algorithm to use to rank pointings.
        Possible options are:

            - ``'probadd'`` : Score by total probability added by each pointing, ignoring previously covered pixels.
            - ``'probden_probadd'`` : Score by maximum probability density in each pointing, breaking ties by total probability added.

    output_dir : str, optional
        The output directory to save results; if not provided, uses the directory the skymap is in.
    n_jobs : int, optional
        Number of parallel workers to use when evaluating footprint coverage per pointing;
        -1 (default) uses all available cores. See joblib.Parallel for details.
    Raises
    ------
    NotImplementedError
        If the scoring specified is not implemented.
    """

    # Load skymap
    print("Loading skymap...")
    skymap_filename, sm = healpix.parse_skymap_args(skymap_filename, lvk_eventname)
    # Load pointings
    print("Loading DECam archival pointings...")
    decam_tiling = decam.get_archival_tiling(force_update=tiling_force_update)
    sc_tiling = SkyCoord(
        decam_tiling["ra"],
        decam_tiling["dec"],
        unit=u.deg,
    )

    # Get pointings within contours
    print("Selecting pointings near 90% contour regions...")
    # # TODO: Back of the envelope calculation shows that 12 hours / (60+30)s * 3 sq. deg. = 1440 square degrees of coverage.
    # #       Based on this, choosing the hpx area to get pointings for should cut off at ~1500 square degrees, or 95%, whichever is smaller
    in_region = healpix.mask_pointings_in_skymap(
        skymap_filename,
        sc_tiling,
        ci=90,
        max_order=11,
    )
    nearby_tiling = decam_tiling[in_region]
    del decam_tiling, sc_tiling

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
    print(
        f"Found {nearby_coverage.shape[0]} clustered pointings near 90% contour regions:"
    )
    for f in decam.FILTERS:
        n_pointings = nearby_coverage[f].sum()
        print(f"\t{f}: {n_pointings} pointings")
    del nearby_tiling, nearby_tiling_skycoord, nearby_tiling_clustered_skycoord
    # Determine coverage of healpixs
    print("Evaluating healpix coverage of pointings with footprint...")
    # Ends up as an [n_pointings x n_healpixels] array, true if that pointing covers the healpixel
    # Get ra/dec for skymap
    sm["RA"], sm["DEC"] = healpix.calc_radecs_for_skymap(sm)
    sc_sm = SkyCoord(
        sm["RA"],
        sm["DEC"],
        unit="deg",
        frame="icrs",
    )
    # Iterate over exposures (in parallel: each pointing's footprint check is independent)
    ra_obj = sc_sm.ra.to(u.deg).value
    dec_obj = sc_sm.dec.to(u.deg).value
    in_footprint = Parallel(n_jobs=n_jobs)(
        delayed(_pointing_in_footprint)(
            pointing.ra, pointing.dec, ra_obj, dec_obj, footprint
        )
        for pointing in nearby_coverage.itertuples()
    )
    # Cast as array
    in_footprint = np.array(in_footprint)
    # Select pointings
    # Get probability contained in hpxs
    hpx_probs = healpix._get_probs_for_skymap(sm)
    # Iterate over filters
    # TODO: The algorithm works, but the output was sloppily put together right before bedtime, so definitely rethink that
    print(f"Selecting obsplan by '{scoring}' scoring...")
    result = {}
    for f in decam.FILTERS:
        i_exps = []
        probs_added = []
        hpx_probs_uncovered = hpx_probs.copy()
        # Iterate until no more coverage is possible
        while True:
            # Calculate probability coverage using only non-covered pixels
            exp_probs = score_by_probadd(hpx_probs_uncovered, in_footprint)
            # Calculate score, mask by filter
            if scoring == "probadd":
                scores = exp_probs
            elif scoring == "probden_probadd":
                scores = score_by_probden_probadd(hpx_probs_uncovered, in_footprint)
            else:
                raise NotImplementedError(f"Scoring '{scoring}' not implemented.")
            scores *= nearby_coverage[f]
            # Break if all scores are 0
            if scores.max() == 0.0:
                break
            # Select exposure covering the most probability
            i_exp = np.argmax(scores)
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

    # Convert pointings to dataframes
    selected_pointings = {}
    for f in decam.FILTERS:
        x = [nearby_coverage.iloc[i]["ra"] for i in result[f]["i_exps"]]
        y = [nearby_coverage.iloc[i]["dec"] for i in result[f]["i_exps"]]
        df = pd.DataFrame({"ra": x, "dec": y, "probs_added": result[f]["probs_added"]})
        selected_pointings[f] = df

    # Package results
    package_results(
        skymap_filename,
        selected_pointings,
        footprint,
        output_dir=output_dir,
    )
