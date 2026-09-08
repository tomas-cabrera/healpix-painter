"""``ZTFFootprint`` and ``ZTFConvexHull`` footprints are currently available, but other information required to run observation planning (e.g. the ZTF tiling) is not implemented.
The footprints can still be used to calculate coverage of ZTF observations, using the :py:func:`healpix_painter.healpix.calc_skymap_coverage` function.
"""

import os.path as pa

import healpix_painter
from healpix_painter.footprints import Footprint

ZTFFootprint = Footprint(
    regions_file=f"{pa.dirname(healpix_painter.__file__)}/data/footprints/ztf.crtf",
    mount="equatorial",
)

ZTFConvexHull = Footprint(
    regions_file=f"{pa.dirname(healpix_painter.__file__)}/data/footprints/ztf.convexhull.crtf",
    mount="equatorial",
)
