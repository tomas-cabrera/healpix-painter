import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion, Regions
import healpix_painter
import os.path as pa


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
        regions : _type_
            _description_
        mount : str, "equatorial" or "altaz"
            Telescope mounting, by default "equatorial"
        """
        if regions_file is not None and region_coords is None:
            self.regions = Regions.read(regions_file)
            self.region_coords = self.region_coords_from_regions()
        elif region_coords is not None and regions_file is None:
            # region_coords is an n_regions x 2 x n_vertices_per_region array
            self.region_coords = jnp.array(region_coords)
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
        region_coords = jnp.einsum(
            "ijk->jik",
            jnp.array(
                [
                    [r.vertices.ra.deg for r in self.regions],
                    [r.vertices.dec.deg for r in self.regions],
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
                        ra=self.region_coords[i, 0, :],
                        dec=self.region_coords[i, 1, :],
                        unit="deg",
                        frame="icrs",
                    )
                )
                for i in range(self.region_coords.shape[0])
            ]
        )
        return regions

    def _footprint_in_cartesian(self):
        # Get the RA and DEC trig values as independent arrays in radians
        _ra, _dec = (
            self.region_coords[:, 0, :] * jnp.pi / 180,
            self.region_coords[:, 1, :] * jnp.pi / 180,
        )
        _rasin, _racos = jnp.sin(_ra), jnp.cos(_ra)
        _decsin, _deccos = jnp.sin(_dec), jnp.cos(_dec)
        # Convert to cartesian coordinates
        _r = jnp.einsum(
            "ijk->jik",
            jnp.array(
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
            jnp.einsum(
                "ijk->jik",
                jnp.array(
                    [
                        jnp.arctan2(_y, _x)
                        + (
                            2 * jnp.pi * (_y < 0)
                        ),  # RA (latter half ensures positive values)
                        jnp.arctan2(_z, jnp.sqrt(_x**2 + _y**2)),  # DEC
                    ]
                ),
            )
            * 180
            / jnp.pi
        )
        return _r

    def _rotate_equatorial(self, ra, dec):
        # Calculate the rotation matrix;
        #   apply dec before ra so that the dec rotation is simply around the negative y-axis.
        #   TODO: add roll, as a rotation about (+/-)x-axis before the ra and dec rotations
        sinra, cosra = jnp.sin(ra * jnp.pi / 180), jnp.cos(ra * jnp.pi / 180)
        sindec, cosdec = jnp.sin(dec * jnp.pi / 180), jnp.cos(dec * jnp.pi / 180)
        ra_matrix = jnp.array(
            [
                [cosra, -sinra, 0],
                [sinra, cosra, 0],
                [0, 0, 1],
            ]
        )
        dec_matrix = jnp.array(
            [
                [cosdec, 0, -sindec],
                [0, 1, 0],
                [sindec, 0, cosdec],
            ]
        )
        rotation_matrix = jnp.linalg.multi_dot([ra_matrix, dec_matrix])
        # Convert footprint to cartesian
        region_coords_cartesian = self._footprint_in_cartesian()
        # Apply rotation matrix
        region_coords_cartesian_rotated = jnp.einsum(
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
        if self.mount == "equatorial":
            return self._rotate_equatorial(ra, dec)
        else:
            raise NotImplementedError(
                f"Mount {self.mount} not implemented; available options are {self._mounts_implemented}."
            )


class _DECamFootprint(Footprint):
    """
    A class to represent a DECam footprint in the sky.
    """

    _footprint_path = (
        f"{pa.dirname(healpix_painter.__file__)}/data/footprints/decam.crtf"
    )

    def __init__(self):
        super().__init__(regions_file=self._footprint_path, mount="equatorial")


DECamFootprint = _DECamFootprint()
