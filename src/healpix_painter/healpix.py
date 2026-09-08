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
from regions import PixCoord, PolygonSkyRegion, Regions

import healpix_painter


def parse_skymap_args(
    skymap_filename=None,
    lvk_eventname=None,
    force_update=False,
    verbose=False,
) -> tuple[str, Table]:
    """Loads skymap from given information, either a local filename or an LIGO/Virgo/KAGRA event name.

    Parameters
    ----------
    skymap_filename : str, optional
        The path to the HEALPix skymap to tile.
        Either skymap_filename or lvk_eventname must be provided.
    lvk_eventname : str, optional
        The LVK event id to tile.
        Either skymap_filename or lvk_eventname must be provided.
    force_update : bool, optional, default False
        If True, forces a re-download of the skymap from GraceDb even if a cached version exists.
    verbose : bool, optional, default False
        If True, prints verbose output.

    Returns
    -------
    tuple[str, astroy.table.Table]
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
    # Get column names
    if isinstance(skymap, Table):
        cols = skymap.columns
    elif isinstance(skymap, np.ndarray):
        cols = skymap.dtype.names
    else:
        raise ValueError("Skymap must be an astropy Table or a numpy structured array.")
    # Get probs
    if "PROB" in cols:
        return skymap["PROB"]
    elif "PROBDENSITY" in cols:
        if "UNIQ" in cols:
            areas = lsm_moc.uniq2pixarea(skymap["UNIQ"])
        else:
            areas = 4 * np.pi / len(skymap)
        return skymap["PROBDENSITY"] * areas
    else:
        raise ValueError("Skymap must have either 'PROB' or 'PROBDENSITY' column.")


def calc_radecs_for_skymap(skymap, flat_order="nested"):
    """Using the UNIQ/HEALPix indexing, calculate the RA and DEC for each pixel center in the skymap.

    Parameters
    ----------
    skymap : astropy.table.Table
        The skymap as an astropy table.
    flat_order : str, optional, default 'nested'
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


def calc_credible_levels_for_skymap(skymap):
    """Calculate the credible levels for each skymap pixel.

    Parameters
    ----------
    skymap : astropy.table.Table
        The skymap as an astropy table.

    Returns
    -------
    np.ndarray
        The credible levels for each pixel, in percent.
    """

    # Get probs
    probs = _get_probs_for_skymap(skymap)

    # Find credible levels
    i = np.flipud(np.argsort(probs))
    cumsum = np.cumsum(probs[i])
    cls = np.empty_like(probs)
    cls[i] = cumsum * 100

    return cls


def calc_contours_for_skymap(skymap_flat, contours):
    """Calculate the contours for the given skymap.

    Parameters
    ----------
    skymap_flat : astropy.table.Table
        The flattened skymap as an astropy table.
    contours : tuple or list
        The contour levels to calculate.

    Returns
    -------
    np.ndarray
        The contours for the given skymap, in the format expected by matplotlib.
    """

    cls = calc_credible_levels_for_skymap(skymap_flat)

    # Generate contours
    # Indexing scheme is paths[CI%][mode][vertex][ra,dec]
    paths = list(postprocess.contour(cls, contours, nest=True, degrees=True))

    return paths


def get_skymap_contours_as_regions(skymap_flat, contours):
    """Convert the contours for the given skymap to regions.

    Parameters
    ----------
    skymap_flat : astropy.table.Table
        The flattened skymap as an astropy table.
    contours : tuple or list
        The contour levels to calculate.

    Returns
    -------
    list of regions
        The contours for the given skymap as regions.
    """

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
    """Given a list of pointings, return a mask on the list indicating which pointings lie in the specified 2D confidence interval.

    Parameters
    ----------
    skymap_path : str
        Path to the HEALPix skymap.
    pointings : SkyCoord
        The pointings to mask.
    ci : int, optional, default 90
        The confidence interval to use, by default 90.
    max_order : int, optional, default 11
        The maximum order to use for the skymap.
    verbose : bool, optional, default False
        If True, prints verbose output.

    Returns
    -------
    np.ndarray
        A boolean mask indicating which pointings lie in the specified confidence interval.
    """

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
    pixcoord = PixCoord.from_sky(pointings, healpix_painter.footprints.DUMMY_WCS)
    in_region = np.zeros(len(pointings), dtype=bool)
    for r90 in r90s:
        pixel_region = r90.to_pixel(healpix_painter.footprints.DUMMY_WCS)
        in_region |= pixel_region.contains(pixcoord)
    if verbose:
        print(sum(in_region), f"exposures in 2D {ci}% confidence interval")
    return in_region


def find_exposures_for_skymap(
    skymap_path: str,
    df_pointings,
    ci=90,
    max_order=11,
    verbose=False,
):
    """
    Find the exposures in a given skymap that fall within a specified confidence interval.

    Parameters
    ----------
    skymap_path : str
        Path to the HEALPix skymap.
    df_pointings : pandas.DataFrame-like
        DataFrame containing the pointings to check.
    ci : int, optional, default 90
        The confidence interval to use.
    max_order : int, optional, default 11
        The maximum order to use for the skymap.
    verbose : bool, optional, default False
        If True, prints verbose output.

    Returns
    -------
    np.ndarray
        A boolean mask indicating which exposures fall within the specified confidence interval.
    """

    # Get coordinates in contour region
    sc = SkyCoord(
        df_pointings["ra_center"],
        df_pointings["dec_center"],
        unit="deg",
    )
    in_region = mask_pointings_in_skymap(skymap_path, sc, ci, max_order)
    if verbose:
        print(sum(in_region), f"exposures in 2D {ci}% confidence interval")
    return in_region


def calc_skymap_coverage(skymap_path: str, pointings_list: list, footprints_list: list):
    """Calculate the skymap probability covered by a set of pointings.
    Pointings for multiple telescopes may be passed.

    Parameters
    ----------
    skymap_path : str
        The local path to a HEALPix skymap.
    pointings_list : list
        A list of length n_telescopes, where each element is an astropy.coordinates.SkyCoord array representing the pointings for that telescope.
    footprints_list : list
        A list of length n_telescopes, where each element is a healpix_painter.Footprint object representing the footprint of that telescope.

    Returns
    -------
    float, list
        The total probability covered by all telescopes, and a list of the probabilities covered by each telescope.
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
        # Append probability covered by this telescope
        prob_coverage.append(np.sum(hpx_probs[in_footprint]))
        # Append mask of hpxs covered by this telescope
        in_footprints.append(in_footprint)
    # Get hpxs covered by any telescope
    in_footprints = np.logical_or.reduce(in_footprints)
    # Calculate total probability covered by all telescopes
    prob_coverage_total = np.sum(hpx_probs[in_footprints])
    return prob_coverage_total, prob_coverage
