import os
import os.path as pa
from urllib.request import urlretrieve

import astropy.units as u
import astropy_healpix as ah
import jax.numpy as jnp
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


def parse_skymap_args(skymap_filename=None, lvk_eventname=None):
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
            pa.dirname(healpix_painter.__file__),
            "data",
            "skymaps",
            ".cache",
            lvk_eventname,
            pa.basename(skymap_url),
        )
        # If file does not exist
        if not pa.exists(skymap_filename):
            # Make directories as needed
            if not pa.exists(pa.dirname(skymap_filename)):
                os.makedirs(pa.dirname(skymap_filename), exist_ok=True)
            # Download file
            print(f"Downloading skymap from {skymap_url} to {skymap_filename}...")
            urlretrieve(skymap_url, skymap_filename)
        else:
            print(f"Using cached skymap at {skymap_filename}...")
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
        healpix_index = jnp.arange(len(skymap))
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
