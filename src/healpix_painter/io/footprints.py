from regions import Regions
import numpy as np


class Footprint(Regions):
    """
    A class to represent a footprint in the sky.
    The footprint should be defined by a list of regions,
    where each region represents a tile in the footprint.
    The coordinates of the region should be defined in the frame where the pointing of the telescope is (ra,dec) = (0,0).
    Rolling/non-equatorial mounts have not been implemented yet.

    Parameters
    ----------
    regions : list of Regions
        A list of regions representing the footprint.
    """

    mounts_implemented = ["equatorial"]

    def __init__(self, regions, mount="equatorial"):
        """_summary_

        Parameters
        ----------
        regions : _type_
            _description_
        mount : str, "equatorial" or "altaz"
            Telescope mounting, by default "equatorial"
        """
        super().__init__(regions)
        self.mount = mount
        assert (
            self.mount in self.mounts_implemented
        ), f"Mount {self.mount} not implemented; available options are {self.mounts_implemented}."

    def _rotate_equatorial(self, ra, dec):
        # Calculate the rotation matrix;
        #   apply dec before ra so that the dec rotation is simply around the negative y-axis.
        sinra, cosra = np.sin(ra), np.cos(ra)
        sindec, cosdec = np.sin(dec), np.cos(dec)
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
        rotation_matrix = ra_matrix @ dec_matrix
        # Convert footprint to cartesian
        # Apply rotation matrix
