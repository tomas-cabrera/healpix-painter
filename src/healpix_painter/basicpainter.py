from .io.healpix import parse_skymap_args


def basic_painter(
    healpixfilename=None,
    lvkeventid=None,
):
    # Load skymap
    sm = parse_skymap_args(healpixfilename, lvkeventid)

    #
