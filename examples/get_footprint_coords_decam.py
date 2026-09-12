import os
import os.path as pa
from urllib.request import urlretrieve

from astropy.io import fits
from astropy.wcs import WCS

import healpix_painter
from healpix_painter.footprints import make_footprint_crtf

# Initialize cached directory
cache_dir = f"{pa.dirname(pa.abspath(__file__))}/.cache/decam"
if not pa.exists(cache_dir):
    os.makedirs(cache_dir)

# Get DECam exposure from NOIRLab AstroArchive
fits_url = (
    "https://astroarchive.noirlab.edu/api/retrieve/8d0a5a25dc24a692f6356684b5b1a7ba/"
)
fits_path = f"{cache_dir}/c4d_231130_004704_ooi_g_v1.fits.fits"
if not pa.exists(fits_path):
    urlretrieve(fits_url, fits_path)

# Get coordinates from header
with fits.open(fits_path) as hdul:
    # Get number of extensions
    n_ext = len(hdul)

    # Get central RA, DEC
    header = hdul[0].header
    centra = header["CENTRA"]
    centdec = header["CENTDEC"]

    # Get corner RAs, DECs
    cornras = []
    corndecs = []
    wcss = []
    for i in range(1, n_ext):
        header = hdul[i].header
        cinds = [1, 2, 4, 3]
        ras = [header[f"COR{j}RA1"] for j in cinds]
        decs = [header[f"COR{j}DEC1"] for j in cinds]
        cornras.append(ras)
        corndecs.append(decs)
        wcss.append(WCS(header))

# Make footprint
footprint_path = f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam.crtf"
footprint_path, footprint = make_footprint_crtf(
    centra,
    centdec,
    cornras,
    corndecs,
    output_path=footprint_path,
)
# Make convex hull footprint
root, ext = pa.splitext(footprint_path)
convexhull_path = f"{root}.convexhull{ext}"
footprint_path, footprint = make_footprint_crtf(
    centra,
    centdec,
    cornras,
    corndecs,
    output_path=convexhull_path,
    convexhull=True,
)
