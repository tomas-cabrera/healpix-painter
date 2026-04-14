import os
import os.path as pa
from glob import glob
from urllib.request import urlretrieve

import astropy.units as u
import astropy_healpix as ah
import ligo.skymap.moc as lsm_moc
import lxml.etree
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from ligo.gracedb.rest import GraceDb
from ligo.skymap import postprocess
from regions import PolygonSkyRegion, Regions

import healpix_painter


def parse_skymap_args(
    skymap_filename=None,
    lvk_eventname=None,
    force_update=False,
    verbose=False,
):
    """Returns skymap as astropy table.
    GraceDb interaction mostly cribbed from gwemopt.io.skymap

    Parameters
    ----------
    skymap_filename : str, optional
        The path to the HEALPix skymap to tile.
        Either skymap_filename or lvk_eventname must be provided.
    lvk_eventname : str, optional
        The LVK event id to tile
        Either skymap_filename or lvk_eventname must be provided.

    Returns
    -------
    str, astropy.table.Table
        The skymap filename and the skymap as an astropy table.

    Raises
    ------
    ValueError
        If either skymap_filename or lvk_eventname are not provided, or both are.
    """
    if skymap_filename is None and lvk_eventname is None:
        raise ValueError("Either skymap_filename or lvk_eventname must be provided.")
    elif skymap_filename is not None and lvk_eventname is not None:
        raise ValueError(
            "Only one of skymap_filename or lvk_eventname should be provided."
        )
    elif skymap_filename is not None:
        pass
    elif lvk_eventname is not None:
        # Check for cached skymap
        skymap_dir = pa.join(
            pa.dirname(healpix_painter.__file__),
            "data",
            "skymaps",
            ".cache",
            lvk_eventname,
        )
        globstr = pa.join(skymap_dir, "*.fits*")
        cached_skymaps = glob(globstr)
        if len(cached_skymaps) > 0 and not force_update:
            skymap_filename = cached_skymaps[0]
            if verbose:
                print(f"Using cached skymap at {skymap_filename}...")
        else:
            if verbose:
                print(f"Fetching skymap for {lvk_eventname} from GraceDb...")
            # Initialize client
            client = GraceDb()
            # Get latest VOEvent info
            latest_voevent = client.voevents(lvk_eventname).json()["voevents"][-1]
            # Get latest skymap url from lxml info
            response = requests.get(latest_voevent["links"]["file"], timeout=60)
            root = lxml.etree.fromstring(response.content)
            params = {
                elem.attrib["name"]: elem.attrib["value"]
                for elem in root.iterfind(".//Param")
            }
            skymap_url = params["skymap_fits"]
            # Make local path
            skymap_filename = pa.join(
                skymap_dir,
                pa.basename(skymap_url),
            )
            # If file does not exist
            if not pa.exists(skymap_filename):
                # Make directories as needed
                if not pa.exists(pa.dirname(skymap_filename)):
                    os.makedirs(pa.dirname(skymap_filename), exist_ok=True)
                # Download file
                if verbose:
                    print(
                        f"Downloading skymap from {skymap_url} to {skymap_filename}..."
                    )
                urlretrieve(skymap_url, skymap_filename)
    return skymap_filename, Table.read(skymap_filename)


def _uniq_to_lonlat(uniq):
    level, ipix = ah.uniq_to_level_ipix(uniq)
    nside = ah.level_to_nside(level)
    lon, lat = ah.healpix_to_lonlat(ipix, nside, order="nested")
    return lon, lat


def _get_probs_for_skymap(skymap):
    try:
        areas = lsm_moc.uniq2pixarea(skymap["UNIQ"])
    except ValueError:
        areas = 4 * np.pi / skymap.shape[0]
    probs = skymap["PROBDENSITY"] * areas
    return probs


def calc_radecs_for_skymap(skymap, flat_order="nested"):
    """Using the UNIQ/HEALPix indexing, calculate the RA and DEC for each pixel in the skymap.

    Parameters
    ----------
    skymap : astropy.table.Table
        The skymap as an astropy table.
    flat_order : str, optional
        The indexing scheme of the flattened skymap; either 'nested' (default) or 'ring'.

    Returns
    -------
    np.ndarray, np.ndarray
        The RA and DEC arrays, in decimal degrees.
    """
    if "UNIQ" in skymap.columns:
        ra, dec = _uniq_to_lonlat(skymap["UNIQ"])
    else:
        healpix_index = np.arange(len(skymap))
        nside = ah.npix_to_nside(len(skymap))
        ra, dec = ah.healpix_to_lonlat(healpix_index, nside, order=flat_order)
    return ra.to(u.deg), dec.to(u.deg)


def calc_contours_for_skymap(skymap_flat, contours):
    # Get probs
    probs = _get_probs_for_skymap(skymap_flat)

    # Find credible levels
    i = np.flipud(np.argsort(probs))
    cumsum = np.cumsum(probs[i])
    cls = np.empty_like(probs)
    cls[i] = cumsum * 100

    # Generate contours
    # Indexing scheme is paths[CI%][mode][vertex][ra,dec]
    paths = list(postprocess.contour(cls, contours, nest=True, degrees=True))

    return paths


def get_skymap_contours_as_regions(skymap_flat, contours):
    # Get contours
    cs = calc_contours_for_skymap(skymap_flat, contours)
    # Convert to regions
    regions = [
        Regions(
            [
                PolygonSkyRegion(
                    vertices=SkyCoord(
                        [v[0] for v in m],
                        [v[1] for v in m],
                        unit="deg",
                        frame="icrs",
                    )
                )
                for m in c
            ]
        )
        for c in cs
    ]
    return regions


def mask_pointings_in_skymap(
    skymap_path: str,
    pointings: SkyCoord,
    ci=90,
    max_order=11,
    verbose=False,
):
    # Get skymap
    _, skymap = parse_skymap_args(skymap_filename=skymap_path)
    # Flatten skymap, capping at max_order to avoid memory overflow
    skymap_flat = lsm_moc.rasterize(
        skymap,
        order=min(
            np.max(lsm_moc.uniq2order(skymap["UNIQ"])),
            max_order,
        ),
    )
    # Get contour regions
    r90s = get_skymap_contours_as_regions(skymap_flat, [ci])[0]

    # Get coordinates in contour region
    in_region = np.array([False] * len(pointings))
    for r90 in r90s:
        in_region_temp = r90.contains(pointings, healpix_painter.footprints.DUMMY_WCS)
        in_region = np.logical_or(in_region, in_region_temp)
    if verbose:
        print(sum(in_region), "exposures in follow-up")
    return in_region


def find_exposures_for_skymap(
    skymap_path: str,
    df_pointings,
    dt_followup=(3 * 365) * u.day,
    ci=90,
    max_order=11,
    verbose=False,
):
    # Get coordinates in contour region
    sc = SkyCoord(
        df_pointings["ra_center"],
        df_pointings["dec_center"],
        unit="deg",
    )
    in_region = mask_pointings_in_skymap(skymap_path, sc, ci, max_order)
    if verbose:
        print(sum(in_region), "exposures in contour region")

    # # Get exposures after event
    # skymap_header = fits.getheader(skymap_path, ext=1)
    # t_event = Time(skymap_header["DATE-OBS"], format="isot")
    # t_exposure = Time(
    #     list(df_pointings["caldat"].apply(lambda x: f"{x}T00:00:00").values),
    #     format="isot",
    # )
    # dt_exposure = t_exposure - t_event
    # after_event = dt_exposure > 0 * u.day
    # within_dt = dt_exposure < dt_followup
    # good_time = np.logical_and(after_event, within_dt)
    # if verbose:
    #     print(sum(good_time), "exposures in follow-up time window")

    # Get exposures in contour region and after event
    # in_followup = np.logical_and(in_region, good_time)
    if verbose:
        print(sum(in_region), "exposures in follow-up")
    return in_region


def calc_skymap_coverage(skymap_path: str, pointings_list: list, footprints_list: list):
    """Calculate the skymap probability covered by a set of pointings.
    Pointings for multiple telescopes may be passed.

    :param skymap_path: The local path to a HEALPix skymap.
    :type skymap_path: str
    :param pointings_list: A list of length n_telescopes, where each element is a list of astropy.coordinates.SkyCoord objects representing the pointings for that telescope.
    :type pointings_list: list
    :param footprints_list: A list of length n_telescopes, where each element is a healpix_painter.Footprint object representing the footprint of that telescope.
    :type footprints_list: list
    :return: The total probability covered by the pointings, and a list of the probabilities covered by each telescope.
    :rtype: float, list
    """
    # Get skymap
    _, skymap = parse_skymap_args(skymap_filename=skymap_path)
    # Get hpx probabilites
    hpx_probs = _get_probs_for_skymap(skymap)
    # Get coords for skymap
    skymap["RA"], skymap["DEC"] = calc_radecs_for_skymap(skymap)
    sc_skymap = SkyCoord(skymap["RA"], skymap["DEC"], unit="deg")
    # Ensure pointings_list and footprints_list are the same length
    assert len(pointings_list) == len(footprints_list), (
        "pointings_list and footprints_list must be the same length."
    )
    # Calculate coverage
    prob_coverage = []
    in_footprints = []
    for pointings, footprint in zip(pointings_list, footprints_list):
        print(f"Calculating coverage for {len(pointings)} pointings...")
        # Mark hpx in pointings
        in_footprint = []
        for pointing in pointings:
            in_pointing = footprint.in_footprint(
                ra_obj=sc_skymap.ra.deg,
                dec_obj=sc_skymap.dec.deg,
                ra_exp=pointing.ra.deg,
                dec_exp=pointing.dec.deg,
            )
            in_footprint.append(in_pointing)
        # Reduce to mask of covered hpxs
        in_footprint = np.logical_or.reduce(in_footprint)
        # Sum probabilities in pointings
        prob_coverage.append(np.sum(hpx_probs[in_footprint]))
        # Append mask of hpxs covered by this telescope
        in_footprints.append(in_footprint)
    # Get hpxs covered by any telescope
    in_footprints = np.logical_or.reduce(in_footprints)
    # Calculate total probability covered by all telescopes
    prob_coverage_total = np.sum(hpx_probs[in_footprints])
    return prob_coverage_total, prob_coverage
