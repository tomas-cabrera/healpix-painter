import os
import os.path as pa
from urllib.request import urlretrieve

from healpix_painter.basicpainter import basic_painter

CACHEDIR = f"{pa.dirname(__file__)}/.cache"
if not pa.exists(CACHEDIR):
    os.makedirs(CACHEDIR)

SKYMAP_URL = (
    # "https://gracedb.ligo.org/api/superevents/S230922g/files/Bilby.multiorder.fits"
    # "https://gracedb.ligo.org/api/superevents/S250727dc/files/Bilby.offline0.multiorder.fits"
    "https://gracedb.ligo.org/api/superevents/S250927ck/files/Bilby.multiorder.fits"
)
SKYMAP_PATH = f"{CACHEDIR}/{pa.basename(SKYMAP_URL)}"


def test_complete():
    # Download sample skymap
    urlretrieve(SKYMAP_URL, SKYMAP_PATH)

    # Run healpix_painter
    basic_painter(skymap_filename=SKYMAP_PATH, scoring="probadd")


if __name__ == "__main__":
    test_complete()
