import os
import os.path as pa

import pandas as pd
import requests

import healpix_painter
from healpix_painter.footprints import Footprint

###############################################################################

TILING_DIR = f"{pa.dirname(pa.dirname(__file__))}/data/tilings/.cache"
os.makedirs(TILING_DIR, exist_ok=True)

FILTERS = list("ugrizY")

DECamFootprint = Footprint(
    regions_file=f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam.crtf",
    mount="equatorial",
)

DECamConvexHull = Footprint(
    regions_file=f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam.convexhull.crtf",
    mount="equatorial",
)


def get_full_archive(force_update=False):
    ARCHIVE_FULL_PATH = f"{TILING_DIR}/archive_full.csv"
    if pa.exists(ARCHIVE_FULL_PATH) and not force_update:
        df = pd.read_csv(ARCHIVE_FULL_PATH)
    else:
        NATROOT = "https://astroarchive.noirlab.edu"
        ADSURL = f"{NATROOT}/api/adv_search"
        print(f"Fetching DECam archival pointings from {NATROOT}...")

        JJ = {
            "outfields": [
                "ra_center",
                "dec_center",
                "ifilter",
                "EXPNUM",
                "exposure",
                "proposal",
                "caldat",
            ],
            "search": [
                ["instrument", "decam"],
                ["obs_type", "object"],
                ["proc_type", "instcal"],
                ["prod_type", "image"],
            ],
        }

        # Fetch
        apiurl = f"{ADSURL}/find/?limit=4000000"
        df = pd.DataFrame(requests.post(apiurl, json=JJ).json()[1:])

        # Save
        df.to_csv(
            ARCHIVE_FULL_PATH,
            index=False,
        )
    return df


def get_archival_tiling(force_update=False):
    ##############################
    ###     Fetch/load df      ###
    ##############################

    # Get archive images
    ARCHIVE_BYFILTER_PATH = f"{TILING_DIR}/archive_byfilter.csv"
    if pa.exists(ARCHIVE_BYFILTER_PATH) and not force_update:
        df_byfilter = pd.read_csv(ARCHIVE_BYFILTER_PATH)
    else:
        # Get full archive
        df = get_full_archive(force_update=force_update)

        # Drop file columns
        file_cols = [col for col in df.columns if "file:" in col]
        df.drop(columns=file_cols, inplace=True)

        # Drop caldat columns
        df.drop(columns=["caldat"], inplace=True)

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

        ##############################
        ###  Add byfilter columns  ###
        ##############################

        # Initialize df_coverage with unique ra/dec of full df
        df.set_index(["ra_center", "dec_center"], inplace=True)
        df_byfilter = pd.DataFrame(index=df.index.unique())
        # Get filter truth columns
        for f in list("ugrizY"):
            df_byfilter[f] = df_byfilter.index.isin(df.index[df["ifilter"] == f])
        # Save
        df_byfilter.reset_index(inplace=True)
        df_byfilter.rename(
            columns={"ra_center": "ra", "dec_center": "dec"},
            inplace=True,
        )
        df_byfilter.to_csv(
            ARCHIVE_BYFILTER_PATH,
            index=False,
        )
        del df

    return df_byfilter
