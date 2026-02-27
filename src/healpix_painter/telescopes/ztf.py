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
