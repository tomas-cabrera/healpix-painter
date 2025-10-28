import os
import os.path as pa
import subprocess
from urllib.request import urlretrieve

from healpix_painter import healpix

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


# def test_gracedb_skymap():
#     skymap = healpix.parse_skymap_args(None, "S241125n")
#     assert skymap is not None
