import os
import os.path as pa
import subprocess
from urllib.request import urlretrieve

from astropy.coordinates import SkyCoord
import pandas as pd

from healpix_painter import healpix
from healpix_painter.telescopes.decam import DECamFootprint
from healpix_painter.telescopes.ztf import ZTFFootprint

# Prep cache
cachedir = pa.join(
    pa.dirname(pa.dirname(pa.dirname(os.path.abspath(__file__)))),
    ".cache",
)
if not pa.exists(cachedir):
    os.makedirs(cachedir)

# Download S230922g skymap for local testing
test_skymap_filename = f"{cachedir}/test_skymap.fits"
if not pa.exists(test_skymap_filename):
    urlretrieve(
        "https://gracedb.ligo.org/api/superevents/S230922g/files/Bilby.multiorder.fits",
        test_skymap_filename,
    )
test_skymap_filename_flattened = test_skymap_filename.replace(
    ".fits", "_flattened.fits"
)
if not pa.exists(test_skymap_filename_flattened):
    subprocess.run(
        [
            "ligo-skymap-flatten",
            test_skymap_filename,
            test_skymap_filename_flattened,
        ]
    )


def test_remote_skymap():
    skymap = healpix.parse_skymap_args(lvk_eventname="S230922g")
    assert skymap is not None


def test_local_skymap():
    skymap = healpix.parse_skymap_args(skymap_filename=test_skymap_filename)
    assert skymap is not None


def test_skymap_coverage():
    # DECam pointings
    df_decam = pd.read_csv("./S251112cm.pointings_decam_g.csv")
    sc_decam = SkyCoord(
        ra=df_decam["ra"].values,
        dec=df_decam["dec"].values,
        unit="deg",
    )
    # ZTF pointings
    df_ztf = pd.read_csv("./S251112cm.pointings_ztf.csv")
    sc_ztf = SkyCoord(
        ra=df_ztf["ra"].values,
        dec=df_ztf["dec"].values,
        unit="deg",
    )
    print("Calculating skymap coverage...")
    prob_coverage_total, prob_coverage = healpix.calc_skymap_coverage(
        "./S251112cm.Bilby.multiorder.fits",
        [sc_decam, sc_ztf],
        [DECamFootprint, ZTFFootprint],
    )
    print("Total probability covered:", prob_coverage_total)
    print("Probability covered by each telescope:", prob_coverage)


if __name__ == "__main__":
    test_remote_skymap()
    test_local_skymap()
    test_skymap_coverage()
