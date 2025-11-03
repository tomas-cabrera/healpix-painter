import os
import os.path as pa

import pandas as pd
import requests

###############################################################################

TILING_DIR = f"{pa.dirname(pa.dirname(__file__))}/data/tilings/.cache"
os.makedirs(TILING_DIR, exist_ok=True)

FILTERS = list("ugrizY")


def get_archival_tiling(force_update=False):
    ##############################
    ###     Fetch/load df      ###
    ##############################

    # Get archive images
    ARCHIVE_BYFILTER_PATH = f"{TILING_DIR}/archive_byfilter.csv"
    if pa.exists(ARCHIVE_BYFILTER_PATH) and not force_update:
        df_byfilter = pd.read_csv(ARCHIVE_BYFILTER_PATH)
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
