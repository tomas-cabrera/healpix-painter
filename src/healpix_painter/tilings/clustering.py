import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky


def assign_groupids(coords, max_sep=1.0 * u.arcmin):
    """Given a set of coordinates, cluster by given radius,
    s.t. the members of each group are within `max_sep` of another member of the group.

    Parameters
    ----------
    skycoords : astropy.coordinates.SkyCoordj
        Set of sky coordinates to cluster.
    max_sep : astropy.coordinates.Angle, optional
        Maximum separation between group member and closest neighbor, by default 1.0*u.arcmin
    """
    # Find nearest neighbors
    idx, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
    # Get chains of neighbors (whose nearest neighbors are within the chain)
    # Iterate over skycoords
    groupids = np.empty(len(coords))
    groupids[:] = np.nan
    groupid = 0
    for i in np.arange(len(coords)):
        # If closer than max_sep
        if d2d[i] <= max_sep:
            skycoord_assigned = not np.isnan(groupids[i])
            neighbor_assigned = not np.isnan(groupids[idx[i]])
            # If neither skycoord nor neighbor assigned
            if not skycoord_assigned and not neighbor_assigned:
                # Assign new groupid and iterate
                groupids[i] = groupid
                groupids[idx[i]] = groupid
                groupid += 1
            # Elif skycoord assigned and neighbor not assigned:
            elif skycoord_assigned and not neighbor_assigned:
                # Carry over groupid
                groupids[idx[i]] = groupids[i]
            # Elif skycoord not assigned and neighbor assigned:
            elif not skycoord_assigned and neighbor_assigned:
                # Carry over groupid
                groupids[i] = groupids[idx[i]]
            # If both skycoord and neighbor assigned
            else:
                # Default to using skycoord groupid
                # Update all neighbor groupids with new groupid
                groupids[groupids == groupids[idx[i]]] = groupids[i]
        else:
            # Assign individual groupid and iterate
            groupids[i] = groupid
            groupid += 1
    # Return groupids
    return groupids


def cluster_once(coords, max_sep=1.0 * u.arcmin):
    """Given a set of coordinates, perform one clustering iteration.
    One clustering iteration finds the nearest neighbors and groups them into a single pointing
    if their separation is less than the threshold.

    Parameters
    ----------
    skycoords : astropy.coordinates.SkyCoordj
        Set of sky coordinates to cluster.
    max_sep : astropy.coordinates.Angle, optional
        Maximum separation between group member and closest neighbor, by default 1.0*u.arcmin
    """
    # Find groupids
    groupids = assign_groupids(coords, max_sep=max_sep)
    # Average coords in group
    outcoords = []
    for gid in np.unique(groupids):
        # Get coords in group
        group_coords = coords[groupids == gid]
        # Calculate spherical offsets from first member to others
        dras, ddecs = group_coords[0].spherical_offsets_to(group_coords)
        # Average offsets
        dra_mean = np.mean(dras)
        ddec_mean = np.mean(ddecs)
        # Offset from first member to get average coord
        group_mean = group_coords[0].spherical_offsets_by(dra_mean, ddec_mean)
        outcoords.append(group_mean)
    return SkyCoord(outcoords)


def cluster_skycoord(coords, max_sep=1.0 * u.arcmin, max_iters=100):
    """Given a set of sky coordinates,
    cluster them s.t. the members of each cluster is within `max_sep` of another member of the cluster.

    Parameters
    ----------
    skycoords : astropy.coordinates.SkyCoordj
        Set of sky coordinates to cluster.
    max_sep : astropy.coordinates.Angle, optional
        Maximum separation between group member and closest neighbor, by default 1.0*u.arcmin
    """
    # Loop until no change occurs
    outcoords_old = []
    outcoords_new = coords.copy()
    iters = 0
    while len(outcoords_old) != len(outcoords_new):
        outcoords_old = outcoords_new.copy()
        outcoords_new = cluster_once(outcoords_new, max_sep=max_sep)
        max_iters += 1
        # Break if max iters reached
        if iters >= max_iters:
            print(
                f"Warning: maximum iterations {max_iters} reached in cluster_skycoord."
            )
            break
    return outcoords_new
