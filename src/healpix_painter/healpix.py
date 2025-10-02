import astropy.units as u
import astropy_healpix as ah
import jax.numpy as jnp
import ligo.skymap.moc as lsm_moc
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from ligo.skymap import postprocess
from regions import PolygonSkyRegion, Regions


def parse_skymap_args(skymap_filename, lvk_eventname):
    if skymap_filename is None and lvk_eventname is None:
        raise ValueError("Either skymap_filename or lvk_eventname must be provided.")
    if skymap_filename is not None and lvk_eventname is not None:
        raise ValueError(
            "Only one of skymap_filename or lvk_eventname should be provided."
        )
    if skymap_filename is not None:
        return Table.read(skymap_filename)
    if lvk_eventname is not None:
        # TODO: download from GraceDB + flatten
        pass


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
    """Returns in radians

    Parameters
    ----------
    skymap : _type_
        _description_
    flat_order : str, optional
        _description_, by default "nested"

    Returns
    -------
    _type_
        _description_
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
