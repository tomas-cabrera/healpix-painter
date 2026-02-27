import os.path as pa

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from regions import PolygonSkyRegion, Regions
from scipy.spatial import ConvexHull

import healpix_painter

# The SkyRegion.contains() method requires a wcs object to be passed,
# which is used to transform both the region and the coordinates to a pixel regime.
# Here we initialize a dummy WCS object for this purpose, à la https://docs.astropy.org/en/stable/wcs/example_create_imaging.html#first-example
DUMMY_WCS = WCS(naxis=2)
DUMMY_WCS.wcs.crpix = [-234.75, 8.3393]
DUMMY_WCS.wcs.cdelt = np.array([-0.066667, 0.066667])
DUMMY_WCS.wcs.crval = [0, -90]
DUMMY_WCS.wcs.ctype = ["RA---AIR", "DEC--AIR"]
DUMMY_WCS.wcs.set_pv([(2, 1, 45.0)])


class Footprint:
    """
    A class to represent a footprint in the sky.

    Parameters
    ----------
    region_coords : list-like
        A list-like containing the region coordinates.
        Each entry in the list should be a 2 x N array-like, where the first row contains the RA for the N region vertices,
        and the second row contains the DEC for the same (both in decimal degrees).
        For the time being we assume that all regions have the same number of vertices, to clean up some matrix operations.
    """

    _mounts_implemented = ["equatorial"]

    def __init__(self, regions_file=None, region_coords=None, mount="equatorial"):
        """_summary_

        Parameters
        ----------
        regions_file : str, optional
            Path to an astropy.regions.Regions file containing the footprint, by default None
            Either regions_file or region_coords must be provided.
        region_coords : list-like, optional
            A list-like containing the region coordinates.
            Each entry in the list should be a 2 x N array-like, where the first row contains the RA for the N region vertices,
            and the second row contains the DEC for the same (both in decimal degrees).
            For the time being we assume that all regions have the same number of vertices, to clean up some matrix operations.
            Either regions_file or region_coords must be provided.
        mount : str, optional
            Telescope mount to use, by default "equatorial"

        Raises
        ------
        ValueError
            If neither regions_file nor region_coords are provided, or both are.
        """
        if regions_file is not None and region_coords is None:
            self.regions = Regions.read(regions_file)
            self.region_coords = self.region_coords_from_regions()
        elif region_coords is not None and regions_file is None:
            # region_coords is an n_regions x 2 x n_vertices_per_region array
            self.region_coords = np.array(region_coords)
            self.regions = self.regions_from_region_coords()
        else:
            raise ValueError("Either region_file or region_coords must be provided.")
        self.mount = mount
        assert self.mount in self._mounts_implemented, (
            f"Mount {self.mount} not implemented; available options are {self._mounts_implemented}."
        )

    def region_coords_from_regions(self, regions=None):
        """
        Convert the regions to region coordinates.
        """
        if regions is None:
            regions = self.regions
        region_coords = np.einsum(
            "ijk->jik",
            np.array(
                [
                    [r.vertices.ra.deg for r in regions],
                    [r.vertices.dec.deg for r in regions],
                ]
            ),
        )
        return region_coords

    def regions_from_region_coords(self, region_coords=None):
        """
        Convert the region coordinates to regions.
        """
        if region_coords is None:
            region_coords = self.region_coords
        regions = Regions(
            [
                PolygonSkyRegion(
                    vertices=SkyCoord(
                        ra=region_coords[i, 0, :],
                        dec=region_coords[i, 1, :],
                        unit="deg",
                        frame="icrs",
                    )
                )
                for i in range(region_coords.shape[0])
            ]
        )
        return regions

    def _footprint_in_cartesian(self):
        # Get the RA and DEC trig values as independent arrays in radians
        _ra, _dec = (
            self.region_coords[:, 0, :] * np.pi / 180,
            self.region_coords[:, 1, :] * np.pi / 180,
        )
        _rasin, _racos = np.sin(_ra), np.cos(_ra)
        _decsin, _deccos = np.sin(_dec), np.cos(_dec)
        # Convert to cartesian coordinates
        _r = np.einsum(
            "ijk->jik",
            np.array(
                [
                    _racos * _deccos,  # x
                    _rasin * _deccos,  # y
                    _decsin,  # z
                ]
            ),
        )
        return _r

    def _footprint_cartesian_to_spherical(self, r_cartesian):
        # r_cartesian should be an array with dimensions n_regions x 3 x n_vertices_per_region
        _x, _y, _z = r_cartesian[:, 0, :], r_cartesian[:, 1, :], r_cartesian[:, 2, :]
        # Convert to spherical coordinates in degrees
        _r = (
            np.einsum(
                "ijk->jik",
                np.array(
                    [
                        np.arctan2(_y, _x)
                        + (
                            2 * np.pi * (_y < 0)
                        ),  # RA (latter half ensures positive values)
                        np.arctan2(_z, np.sqrt(_x**2 + _y**2)),  # DEC
                    ]
                ),
            )
            * 180
            / np.pi
        )
        return _r

    def _rotate_equatorial(self, ra, dec):
        # Calculate the rotation matrix;
        #   apply dec before ra so that the dec rotation is simply around the negative y-axis.
        #   TODO: add roll, as a rotation about (+/-)x-axis before the ra and dec rotations
        sinra, cosra = np.sin(ra * np.pi / 180), np.cos(ra * np.pi / 180)
        sindec, cosdec = np.sin(dec * np.pi / 180), np.cos(dec * np.pi / 180)
        ra_matrix = np.array(
            [
                [cosra, -sinra, 0],
                [sinra, cosra, 0],
                [0, 0, 1],
            ]
        )
        dec_matrix = np.array(
            [
                [cosdec, 0, -sindec],
                [0, 1, 0],
                [sindec, 0, cosdec],
            ]
        )
        rotation_matrix = np.linalg.multi_dot([ra_matrix, dec_matrix])
        # Convert footprint to cartesian
        region_coords_cartesian = self._footprint_in_cartesian()
        # Apply rotation matrix
        region_coords_cartesian_rotated = np.einsum(
            "ij,kjl->kil",
            rotation_matrix,
            region_coords_cartesian,
        )
        # Convert to spherical
        region_coords_spherical_rotated = self._footprint_cartesian_to_spherical(
            region_coords_cartesian_rotated
        )
        return region_coords_spherical_rotated

    def rotate(self, ra, dec):
        """Returns the region coords of the footprint rotated to the specified coordinates.

        Parameters
        ----------
        ra : _type_
            Right ascension, in decimal degrees.
        dec : _type_
            Declination, in decimal degrees

        Returns
        -------
        np.ndarray
            The rotated region coordinates, as an n_regions x 2 x n_vertices_per_region array.

        Raises
        ------
        NotImplementedError
            If the mount specified is not implemented.
        """
        if self.mount == "equatorial":
            return self._rotate_equatorial(ra, dec)
        else:
            raise NotImplementedError(
                f"Mount {self.mount} not implemented; available options are {self._mounts_implemented}."
            )

    def in_footprint(self, ra_obj, dec_obj, ra_exp=None, dec_exp=None):
        """Returns true if the obj is in the footprint, false otherwise.

        Parameters
        ----------
        ra_obj : _type_
            RA of the object, in decimal degrees.
            Accepts arrays.
        dec_obj : _type_
            Dec of the object, in decimal degrees.
            Accepts arrays.
        ra_exp : _type_, optional
            RA of the exposure, in decimal degrees, by default None.
            If not none, dec_exp must be specified.
        dec_exp : _type_, optional
            Dec of the exposure, in decimal degrees, by default None
            If not none, ra_exp must be specified.
        """
        # Check exposure coords
        if ra_exp is not None and dec_exp is not None:
            pass
        elif ra_exp is None and dec_exp is None:
            ra_exp = 0.0
            dec_exp = 0.0
        else:
            raise ("Both ra_exp and dec_exp must be specified, or neither.")
        # Get rotated region
        rotated_regions = self.regions_from_region_coords(
            region_coords=self.rotate(ra_exp, dec_exp)
        )
        # Check if coords are in regions
        sc_obj = SkyCoord(ra=ra_obj, dec=dec_obj, unit=u.deg)
        obj_in_footprint = []
        for r in rotated_regions:
            obj_in_footprint.append(r.contains(sc_obj, DUMMY_WCS))
        obj_in_footprint = np.array(obj_in_footprint).any(axis=0)
        return obj_in_footprint


def make_footprint_crtf(
    centra,
    centdec,
    cornras,
    corndecs,
    output_path=None,
    convexhull=False,
):
    """Function to convert coordinates from a sample exposure into a CRTF file representing the footprint.
    The required information is the ra/dec of the exposure itself, and the coordinates of the individual CCDs.
    cornras[i][j] should represent the ra of the jth corner of the ith CCD, and similarly for corndecs.
    Also has option to save convexhull of footprint for more lightweight operations.

    :param centra: Central RA of the footprint
    :type centra: float
    :param centdec: Central Dec of the footprint
    :type centdec: float
    :param cornras: Corner RAs of the footprint
    :type cornras: array-like
    :param corndecs: Corner Decs of the footprint
    :type corndecs: array-like
    :param output_path: Path to save the CRTF file, defaults to None
    :type output_path: str, optional
    :param convexhull: Save the convex hull of the footprint, defaults to False
    :type convexhull: bool, optional
    :return: None
    :rtype: None
    """
    # Rotate coords to frame centered on (0,0)
    # Same as footprints.Footprint._rotate_equatorial, but in reverse
    # Rotation matrix
    sinra, cosra = np.sin(centra * np.pi / 180), np.cos(centra * np.pi / 180)
    sindec, cosdec = np.sin(centdec * np.pi / 180), np.cos(centdec * np.pi / 180)
    ra_matrix = np.array(
        [
            [cosra, sinra, 0],
            [-sinra, cosra, 0],
            [0, 0, 1],
        ]
    )
    dec_matrix = np.array(
        [
            [cosdec, 0, sindec],
            [0, 1, 0],
            [-sindec, 0, cosdec],
        ]
    )
    rotation_matrix = np.linalg.multi_dot([dec_matrix, ra_matrix])
    # Flatten ras and decs for matrix multiplication
    ras_np = np.array(cornras)
    decs_np = np.array(corndecs)
    ras_shape = ras_np.shape
    decs_shape = decs_np.shape
    ras_flat = ras_np.reshape(-1)
    decs_flat = decs_np.reshape(-1)
    # Convert to cartesian coordinates
    sinras_flat, cosras_flat = (
        np.sin(ras_flat * np.pi / 180),
        np.cos(ras_flat * np.pi / 180),
    )
    sindecs_flat, cosdecs_flat = (
        np.sin(decs_flat * np.pi / 180),
        np.cos(decs_flat * np.pi / 180),
    )
    r_cartesian_flat = np.array(
        [
            cosras_flat * cosdecs_flat,  # x
            sinras_flat * cosdecs_flat,  # y
            sindecs_flat,  # z
        ]
    )
    # Apply rotation matrix
    r_cartesian_flat_rotated = np.einsum(
        "ij,jk->ik",
        rotation_matrix,
        r_cartesian_flat,
    )
    # Convert to ra/dec
    x_flat_rotated = r_cartesian_flat_rotated[0, :]
    y_flat_rotated = r_cartesian_flat_rotated[1, :]
    z_flat_rotated = r_cartesian_flat_rotated[2, :]
    ras_rotated_flat = (
        (
            np.arctan2(y_flat_rotated, x_flat_rotated)
            + (2 * np.pi * (y_flat_rotated < 0))
        )
        * 180
        / np.pi
    )
    decs_rotated_flat = (
        np.arctan2(z_flat_rotated, np.sqrt(x_flat_rotated**2 + y_flat_rotated**2))
        * 180
        / np.pi
    )

    # If convexhull, get the convex hull of the footprint
    if convexhull:
        # Modification to ras avoids having to deal with meridian
        # TODO: Make this smarter by checking if the footprint crosses the meridian first
        hull = ConvexHull(
            np.array([(ras_rotated_flat + 180) % 360, decs_rotated_flat]).T
        )
        ras_hull = ras_rotated_flat[hull.vertices]
        decs_hull = decs_rotated_flat[hull.vertices]

        # Format for footprint
        footprint_coords = np.array([[ras_hull, decs_hull]])
    # Else keep all details
    else:
        # Reshape back to original shape
        cornras_rotated = ras_rotated_flat.reshape(ras_shape)
        corndecs_rotated = decs_rotated_flat.reshape(decs_shape)
        # Format for footprint
        footprint_coords = np.array(
            [
                [cornras_rotated[i, :], corndecs_rotated[i, :]]
                for i in range(len(cornras_rotated))
            ]
        )

    # Cast as footprint
    footprint = Footprint(region_coords=footprint_coords)

    # Save to CRTF file
    if output_path is None:
        output_path = (
            f"{pa.dirname(healpix_painter.__file__)}/data/footprints/footprint.crtf"
        )
    footprint.regions.write(output_path, overwrite=True)

    return output_path, footprint
