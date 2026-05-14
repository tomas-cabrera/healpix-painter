import os
import os.path as pa

import pandas as pd
from astropy.coordinates import SkyCoord
from pytest import approx

from healpix_painter import healpix
from healpix_painter.telescopes.decam import DECamFootprint
from healpix_painter.telescopes.ztf import ZTFFootprint

# Get test data directory
datadir = pa.join(
    pa.dirname(pa.dirname(os.path.abspath(__file__))),
    "test_data",
)

# File paths
S230922g_skymap_filename = pa.join(
    datadir,
    "S230922g.Bilby.multiorder.fits",
)
S251112cm_skymap_filename = pa.join(
    datadir,
    "S251112cm.Bilby.multiorder.fits",
)
S25111cm_pointings_decam_filename = pa.join(
    datadir,
    "S251112cm.pointings_decam.csv",
)
S25111cm_pointings_ztf_filename = pa.join(
    datadir,
    "S251112cm.pointings_ztf.csv",
)


def test_remote_skymap():
    skymap = healpix.parse_skymap_args(lvk_eventname="S230922g")
    assert skymap is not None


def test_local_skymap():
    skymap = healpix.parse_skymap_args(skymap_filename=S230922g_skymap_filename)
    assert skymap is not None


def test_skymap_coverage():
    # DECam pointings
    df_decam = pd.read_csv(S25111cm_pointings_decam_filename)
    sc_decam = SkyCoord(
        ra=df_decam["ra"].values,
        dec=df_decam["dec"].values,
        unit="deg",
    )
    # ZTF pointings
    df_ztf = pd.read_csv(S25111cm_pointings_ztf_filename)
    sc_ztf = SkyCoord(
        ra=df_ztf["ra"].values,
        dec=df_ztf["dec"].values,
        unit="deg",
    )
    print("Calculating skymap coverage...")
    prob_coverage_total, prob_coverage = healpix.calc_skymap_coverage(
        S251112cm_skymap_filename,
        [sc_decam, sc_ztf],
        [DECamFootprint, ZTFFootprint],
    )
    print("Total probability covered:", prob_coverage_total)
    print("Probability covered by each telescope:", prob_coverage)
    assert prob_coverage_total == approx(0.5637, abs=1e-4)
    assert prob_coverage == approx([0.1955, 0.4249], abs=1e-4)


if __name__ == "__main__":
    test_remote_skymap()
    test_local_skymap()
    test_skymap_coverage()
