import astropy.units as u
import ligo.skymap.moc as lsm_moc
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky

from healpix_painter import healpix
from healpix_painter.footprints import DUMMY_WCS, DECamConvexHull
from healpix_painter.io.output import package_results
from healpix_painter.tilings import decam
from healpix_painter.tilings.clustering import cluster_skycoord


def score_by_probadd(hpx_probs, in_footprint):
    # Calculate probability coverage using only non-covered pixels
    exp_probs = (in_footprint * hpx_probs).sum(axis=1)
    # Return
    return exp_probs


def score_by_probden_probadd(hpx_probs, in_footprint):
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
):
    """Tile the skymap using the given footprint and the given scoring algorithm.
    Currently uses DECam archival pointings.

    Parameters
    ----------
    skymap_filename : str, optional
        The path to the HEALPix skymap to tile.
        Either skymap_filename or lvk_eventname must be provided.
    lvk_eventname : str, optional
        The LVK event id to tile
        Either skymap_filename or lvk_eventname must be provided.
    footprint : healpix_painter.footprints.Footprint, optional
        The telescope footprint to use for tiling; DECamConvexHull by default.
    tiling_force_update : bool, optional
        Whether to force update the tiling cache; False by default.
    max_sep_cluster : astropy.coordinates.Angle, optional
        The radius to use when clustering pointings, by default 1.0*u.arcmin
    scoring : str, optional
        The scoring algorithm to use to rank pointings; 'probadd' by default.
        Possible options are:
        - 'probadd': Score by total probability added by each pointing, ignoring previously covered pixels.
        - 'probden_probadd': Score by maximum probability density in each pointing, breaking ties by total probability added.
    output_dir : str, optional
        The output directory to save results; if not provided, uses the directory the skymap is in.

    Raises
    ------
    NotImplementedError
        If the scoring specified is not implemented.
    """
    # Load skymap
    print("Loading skymap...")
    skymap_filename, sm = healpix.parse_skymap_args(skymap_filename, lvk_eventname)
    # Calculate contour regions
    # Flatten skymap
    sm_flat = lsm_moc.rasterize(sm)
    # Get 90% contour regions
    print("Calculating 90% contour regions...")
    r90s = healpix.get_skymap_contours_as_regions(sm_flat, [90])[0]
    # Load pointings
    print("Loading DECam archival pointings...")
    decam_tiling = decam.get_archival_tiling(force_update=tiling_force_update)
    # Get pointings within contours
    print("Selecting pointings near 90% contour regions...")
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
