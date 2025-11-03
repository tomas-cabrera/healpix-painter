import os
import os.path as pa
import subprocess

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy_healpix as ah
import multiprocessing as mp
import parmap

###############################################################################

NATROOT = "https://astroarchive.noirlab.edu"
ADSURL = f"{NATROOT}/api/adv_search"

JJ = {
    "outfields": [
        "ra_center",
        "dec_center",
        "ifilter",
        "EXPNUM",
        "exposure",
        "proposal",
    ],
    "search": [
        ["instrument", "decam"],
        ["obs_type", "object"],
        ["proc_type", "instcal"],
        ["prod_type", "image"],
    ],
}

##############################
###     Fetch/load df      ###
##############################

# Get archive images
PROJ_PATH = "/hildafs/project/phy220048p/tcabrera/decam_followup_O4/DECam_coverage"
ARCHIVE_IMAGES_PATH = f"{pa.dirname(__file__)}/.cache/archive_coverage.csv"
if pa.exists(ARCHIVE_IMAGES_PATH):
    df = pd.read_csv(ARCHIVE_IMAGES_PATH)
else:
    # Fetch
    apiurl = f"{ADSURL}/find/?limit=4000000"
    df = pd.DataFrame(requests.post(apiurl, json=JJ).json()[1:])

    # Reduce by filter
    df["ifilter"] = df["ifilter"].apply(lambda x: "NaN" if x is None else x[0])
    df = df[df["ifilter"].apply(lambda x: x in list("ugrizY"))]

    # Reduce by exposure times
    df.drop(
        index=df.index[(df["ifilter"] == "g") & (df["exposure"] < 30)], inplace=True
    )
    df.drop(
        index=df.index[(df["ifilter"] != "g") & (df["exposure"] < 50)], inplace=True
    )

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Save
    df.to_csv(ARCHIVE_IMAGES_PATH, index=False)

##############################
###  Add byfilter columns  ###
##############################

# Initialize df_coverage with unique ra/dec of full df
df.set_index(["ra_center", "dec_center"], inplace=True)
df_byfilter = pd.DataFrame(index=df.index.unique())
# Get filter truth columns
for f in list("ugrizY"):
    df_byfilter[f] = df_byfilter.index.isin(df.index[df["ifilter"] == f])
# Get proposal (propid) truth columns
for p in ["2012B-0001", "2014B-0404", "2019A-0305"]:
    df_byfilter[p] = df_byfilter.index.isin(df.index[df["proposal"] == p])
# Save
df_byfilter.reset_index(inplace=True)
df_byfilter.to_csv(f"{pa.dirname(__file__)}/archive_byfilter.csv")
del df

##############################
### Cluster pointings ###
##############################

# SkyCoord of archive
sc_archive = SkyCoord(
    ra=df_byfilter["ra_center"],
    dec=df_byfilter["dec_center"],
    unit=u.deg,
)
# Define HEALPix regions
# Regions are defined as level-3 HEALPix tiles, plus the tolerances to ensure boundaries are non-issues
N_PROC = 16
HPX_LEVEL = 3
MAX_SEP = 60 * u.arcsec


def calc_neighbors_for_healpix(i_hpx):
    """_summary_
    This function calculates the neighborhood relations for one HEALPix tile of a given index and order.
    Doing this in ring because it's entirely internal and requires less lines.

    Parameters
    ----------
    i_hpx : int
        The HEALPix index of the tile to process.
    """
    # Get hpx vertices
    hpx_corners = ah.boundaries_lonlat(
        i_hpx,
        step=1,
        nside=ah.level_to_nside(HPX_LEVEL),
    )
    # Enlarge by MAX_SEP, along line from center
    hpx_center = ah.healpix_to_lonlat(
        i_hpx,
        nside=ah.level_to_nside(HPX_LEVEL),
    )
    sc_corners = SkyCoord(ra=hpx_corners[0], dec=hpx_corners[1])
    sc_center = SkyCoord(ra=hpx_center[0], dec=hpx_center[1])
    print(i_hpx, sc_center, sc_corners, "\n")
    # TODO: get unit vector from center to each corner and offset each corner by MAX_SEP


# Process by hpx regions
i_hpxs = np.arange(ah.nside_to_npix(ah.level_to_nside(HPX_LEVEL)))
with mp.Pool(N_PROC) as pool:
    parmap.map(
        calc_neighbors_for_healpix,
        i_hpxs,
        pm_pool=pool,
        pm_pbar=True,
    )
