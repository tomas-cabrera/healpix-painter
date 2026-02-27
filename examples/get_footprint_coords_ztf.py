import os
import os.path as pa
from urllib.request import urlretrieve

import ztfquery
from astropy.io import fits
from astropy.wcs import WCS

import healpix_painter
from healpix_painter.footprints import make_footprint_crtf

# Initialize cache directory
cache_dir = f"{pa.dirname(pa.abspath(__file__))}/.cache/ztf"
if not pa.exists(cache_dir):
    os.makedirs(cache_dir)

# Get ZTF focal plane coords from image
# The image is saved on the server as quadrants of CCDs:
# There are 16 CCDs, each with 4 quadrants, for a total of 64 quadrants.
# One corner is taken from each quadrant to get the CCD footprint.
filename_template = "ztf_20180525484722_000600_zg_c<CCDID>_o_q<QID>_sciimg.fits"
cornras = []
corndecs = []
wcss = []
centra = None
centdec = None
# Iterate over CCDs
for ccdid in range(1, 17):
    # Initialize list of ras, decs for this CCD
    ras = []
    decs = []
    # Iterate over quadrants
    for qid in range(1, 5):
        # Get file from IRSA
        basename = filename_template.replace(
            "<CCDID>",
            f"{ccdid:02d}",
        ).replace(
            "<QID>",
            f"{qid}",
        )
        fits_path = f"{cache_dir}/{basename}"
        if not pa.exists(fits_path):
            fits_url = ztfquery.buildurl.filename_to_url(basename)
            urlretrieve(fits_url, fits_path)
        # Get info from header
        with fits.open(fits_path) as hdul:
            # Save central RA, DEC (should be identical across quadrants)
            if centra is None and centdec is None:
                centra = hdul[0].header["RAD"]
                centdec = hdul[0].header["DECD"]
            # Get WCS
            wcs = WCS(hdul[0].header)
            wcss.append(wcs)
            # Get number of pixels in each dimension
            naxis1 = hdul[0].header["NAXIS1"]
            naxis2 = hdul[0].header["NAXIS2"]
        # Get corner coords (only one per qid)
        if qid == 1:
            corner = [0, 0]
        elif qid == 2:
            corner = [naxis1, 0]
        elif qid == 3:
            corner = [naxis1, naxis2]
        elif qid == 4:
            corner = [0, naxis2]
        # Convert pixel coords to RA, DEC
        ra, dec = wcs.wcs_pix2world(corner[0], corner[1], 0)
        # Append to list for CCD
        ras.append(ra)
        decs.append(dec)
    # Append to list for focal plane
    cornras.append(ras)
    corndecs.append(decs)

# Make footprint
footprint_path = f"{pa.dirname(healpix_painter.__file__)}/data/footprints/ztf.crtf"
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

# Plot
# import matplotlib.pyplot as plt
# print(cornras)
# print(corndecs)
# plt.plot(
#     centra + 360 if centra < 180 else centra,
#     centdec,
#     "o",
# )
# for ras, decs in zip(cornras, corndecs):
#     plt.plot(
#         [ra + 360 if ra < 180 else ra for ra in ras],
#         decs,
#     )
# plt.show()
# plt.close()
