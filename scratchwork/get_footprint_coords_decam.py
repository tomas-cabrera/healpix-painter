import os.path as pa
from urllib.request import urlretrieve

import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import ConvexHull

import healpix_painter
from healpix_painter import footprints

fits_url = (
    "https://astroarchive.noirlab.edu/api/retrieve/8d0a5a25dc24a692f6356684b5b1a7ba/"
)
fits_path = f"{pa.dirname(pa.abspath(__file__))}/c4d_231130_004704_ooi_g_v1.fits.fits"
if not pa.exists(fits_path):
    urlretrieve(fits_url, fits_path)

# Get central RA, DEC
with fits.open(fits_path) as hdul:
    # Get number of extensions
    n_ext = len(hdul)

    # Get central RA, DEC
    header = hdul[0].header
    ra = header["CENTRA"]
    dec = header["CENTDEC"]

    # Get corner RA, DEC
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

# Rotate coords to frame centered on (0,0)
# Same as footprints.Footprint._rotate_equatorial, but in reverse
# Rotation matrix
sinra, cosra = jnp.sin(ra * jnp.pi / 180), jnp.cos(ra * jnp.pi / 180)
sindec, cosdec = jnp.sin(dec * jnp.pi / 180), jnp.cos(dec * jnp.pi / 180)
ra_matrix = jnp.array(
    [
        [cosra, sinra, 0],
        [-sinra, cosra, 0],
        [0, 0, 1],
    ]
)
dec_matrix = jnp.array(
    [
        [cosdec, 0, sindec],
        [0, 1, 0],
        [-sindec, 0, cosdec],
    ]
)
rotation_matrix = jnp.linalg.multi_dot([dec_matrix, ra_matrix])
# Flatten ras and decs for matrix multiplication
ras_jnp = jnp.array(cornras)
decs_jnp = jnp.array(corndecs)
ras_shape = ras_jnp.shape
decs_shape = decs_jnp.shape
ras_flat = ras_jnp.reshape(-1)
decs_flat = decs_jnp.reshape(-1)
# Convert to cartesian coordinates
sinras_flat, cosras_flat = (
    jnp.sin(ras_flat * jnp.pi / 180),
    jnp.cos(ras_flat * jnp.pi / 180),
)
sindecs_flat, cosdecs_flat = (
    jnp.sin(decs_flat * jnp.pi / 180),
    jnp.cos(decs_flat * jnp.pi / 180),
)
r_cartesian_flat = jnp.array(
    [
        cosras_flat * cosdecs_flat,  # x
        sinras_flat * cosdecs_flat,  # y
        sindecs_flat,  # z
    ]
)
# Apply rotation matrix
r_cartesian_flat_rotated = jnp.einsum(
    "ij,jk->ik",
    rotation_matrix,
    r_cartesian_flat,
)
# Convert to ra/dec
x_flat_rotated = r_cartesian_flat_rotated[0, :]
y_flat_rotated = r_cartesian_flat_rotated[1, :]
z_flat_rotated = r_cartesian_flat_rotated[2, :]
ras_rotated_flat = (
    (jnp.arctan2(y_flat_rotated, x_flat_rotated) + (2 * jnp.pi * (y_flat_rotated < 0)))
    * 180
    / jnp.pi
)
decs_rotated_flat = (
    jnp.arctan2(z_flat_rotated, jnp.sqrt(x_flat_rotated**2 + y_flat_rotated**2))
    * 180
    / jnp.pi
)
# Reshape back to original shape
cornras_rotated = ras_rotated_flat.reshape(ras_shape)
corndecs_rotated = decs_rotated_flat.reshape(decs_shape)

# Convert to healpix_painter.io.footprint.Footprint to get regions object + write
region_coords_rotated = jnp.array(
    [[cornras_rotated[i, :], corndecs_rotated[i, :]] for i in range(n_ext - 1)]
)
footprint = footprints.Footprint(region_coords=region_coords_rotated)
footprint_path = f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam.crtf"
footprint.regions.write(footprint_path, overwrite=True)
footprint_read = footprints.Footprint(regions_file=footprint_path)
cornras_rotated_read = jnp.array(footprint_read.region_coords[:, 0, :])
corndecs_rotated_read = jnp.array(footprint_read.region_coords[:, 1, :])
# These should be roughly identical (entries should be ~0)
coord_errors = footprint_read.region_coords - footprint.region_coords
tol = 0.25
print(
    f'After reload of saved footprint, median coordinate error is {jnp.median(coord_errors) * 3600:.3f}", with {jnp.sum(coord_errors > tol / 3600)}/{jnp.prod(jnp.array(coord_errors.shape))} coords off by >{tol}" arcsec'
)

# from footprints import DECamFootprint
# print(DECamFootprint.region_coords - footprint.region_coords)

# import matplotlib.pyplot as plt
# ax = plt.subplot()
# # We can plot the regions, but they need to be converted to jnp.pixel coordinates first
# # To make the CCDs plot with the correct position w.r.t. one another, a specific WCS needs to be chosen.
# # Because it's not clear what the definition of the centra/dec are,
# # perhaps we should use a CCD (e.g. the first one) as the reference frame.
# # Being consistent across telescopes could be difficult.
# for i in range(n_ext-1):
#     # plt.plot(
#     #     [r - 360 if r > 180 else r for r in cornras[i]],
#     #     corndecs[i]
#     # )
#     jnp.pixel_region = regions[i].to_jnp.pixel(wcss[0])
#     jnp.pixel_region.plot()
# plt.show()
# plt.close()

### Convex hull

ras_temp = ras_rotated_flat
decs_temp = decs_rotated_flat

# Modification to ras avoids having to deal with meridian
hull = ConvexHull(jnp.array([(ras_temp + 180) % 360, decs_temp]).T)
ras_hull = ras_temp[hull.vertices]
decs_hull = decs_temp[hull.vertices]

# Package into region and save
hull_coords = jnp.array([[ras_hull, decs_hull]])
hullprint = footprints.Footprint(region_coords=hull_coords)
hullprint_path = (
    f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam_convexhull.crtf"
)
hullprint.regions.write(hullprint_path, overwrite=True)
